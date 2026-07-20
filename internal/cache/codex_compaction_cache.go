package cache

import (
	"crypto/sha256"
	"encoding/hex"
	"sort"
	"strings"
	"sync"
	"time"
)

const (
	// CodexCompactionCacheTTL limits how long compaction checkpoints stay in
	// process memory.
	CodexCompactionCacheTTL = 1 * time.Hour

	// CodexCompactionCacheMaxEntries bounds process memory for compaction
	// continuity. Oldest entries are evicted first.
	CodexCompactionCacheMaxEntries = 4096

	// CodexCompactionMaxCheckpointsPerEntry bounds retained checkpoints for one
	// session so older prefixes stay matchable after client-side history edits.
	CodexCompactionMaxCheckpointsPerEntry = 8

	// CodexCompactionCacheEvictBatchSize leaves headroom after the cache reaches
	// capacity so high write volume does not rescan the map every turn.
	CodexCompactionCacheEvictBatchSize = 64
)

// CodexCompactionCheckpoint records one server-side compaction result: the
// hashes of the translated input items the compaction covered and the
// replacement history returned by the upstream compact endpoint.
//
// Checkpoints are process-local; in home/multi-node mode a miss simply causes
// the request to be compacted again upstream.
type CodexCompactionCheckpoint struct {
	PrefixHashes []string
	Replacement  [][]byte
}

type codexCompactionEntry struct {
	Checkpoints []CodexCompactionCheckpoint
	Timestamp   time.Time
}

var (
	codexCompactionMu      sync.Mutex
	codexCompactionEntries = make(map[string]codexCompactionEntry)
)

// HashCodexCompactionItem returns the canonical hash used to fingerprint one
// translated Codex input item for compaction prefix matching.
func HashCodexCompactionItem(item []byte) string {
	sum := sha256.Sum256(item)
	return hex.EncodeToString(sum[:])
}

// CacheCodexCompactionCheckpoint stores a compaction checkpoint for a session.
// The newest checkpoint is kept first so longest-prefix matches win.
func CacheCodexCompactionCheckpoint(modelName, sessionKey string, checkpoint CodexCompactionCheckpoint) bool {
	key := codexCompactionCacheKey(modelName, sessionKey)
	if key == "" || len(checkpoint.PrefixHashes) == 0 || len(checkpoint.Replacement) == 0 {
		return false
	}
	cacheCleanupOnce.Do(startCacheCleanup)
	now := time.Now()
	codexCompactionMu.Lock()
	defer codexCompactionMu.Unlock()
	entry := codexCompactionEntries[key]
	if now.Sub(entry.Timestamp) > CodexCompactionCacheTTL {
		entry.Checkpoints = nil
	}
	entry.Checkpoints = append([]CodexCompactionCheckpoint{cloneCodexCompactionCheckpoint(checkpoint)}, entry.Checkpoints...)
	if len(entry.Checkpoints) > CodexCompactionMaxCheckpointsPerEntry {
		entry.Checkpoints = entry.Checkpoints[:CodexCompactionMaxCheckpointsPerEntry]
	}
	entry.Timestamp = now
	codexCompactionEntries[key] = entry
	if len(codexCompactionEntries) > CodexCompactionCacheMaxEntries {
		evictOldestCodexCompactionEntries(CodexCompactionCacheEvictBatchSize)
	}
	return true
}

// MatchCodexCompactionCheckpoint returns the newest checkpoint whose prefix
// hashes exactly match the head of itemHashes.
func MatchCodexCompactionCheckpoint(modelName, sessionKey string, itemHashes []string) (CodexCompactionCheckpoint, bool) {
	key := codexCompactionCacheKey(modelName, sessionKey)
	if key == "" || len(itemHashes) == 0 {
		return CodexCompactionCheckpoint{}, false
	}
	cacheCleanupOnce.Do(startCacheCleanup)
	now := time.Now()
	codexCompactionMu.Lock()
	defer codexCompactionMu.Unlock()
	entry, ok := codexCompactionEntries[key]
	if !ok {
		return CodexCompactionCheckpoint{}, false
	}
	if now.Sub(entry.Timestamp) > CodexCompactionCacheTTL {
		delete(codexCompactionEntries, key)
		return CodexCompactionCheckpoint{}, false
	}
	for _, checkpoint := range entry.Checkpoints {
		if codexCompactionPrefixMatches(checkpoint.PrefixHashes, itemHashes) {
			entry.Timestamp = now
			codexCompactionEntries[key] = entry
			return cloneCodexCompactionCheckpoint(checkpoint), true
		}
	}
	return CodexCompactionCheckpoint{}, false
}

// DeleteCodexCompactionCheckpoints removes all checkpoints for a session, e.g.
// after upstream rejects a substituted history as stale.
func DeleteCodexCompactionCheckpoints(modelName, sessionKey string) {
	key := codexCompactionCacheKey(modelName, sessionKey)
	if key == "" {
		return
	}
	codexCompactionMu.Lock()
	delete(codexCompactionEntries, key)
	codexCompactionMu.Unlock()
}

// ClearCodexCompactionCache clears all compaction checkpoints.
func ClearCodexCompactionCache() {
	codexCompactionMu.Lock()
	codexCompactionEntries = make(map[string]codexCompactionEntry)
	codexCompactionMu.Unlock()
}

func codexCompactionPrefixMatches(prefixHashes, itemHashes []string) bool {
	if len(prefixHashes) == 0 || len(prefixHashes) > len(itemHashes) {
		return false
	}
	for i := range prefixHashes {
		if prefixHashes[i] != itemHashes[i] {
			return false
		}
	}
	return true
}

func cloneCodexCompactionCheckpoint(checkpoint CodexCompactionCheckpoint) CodexCompactionCheckpoint {
	cloned := CodexCompactionCheckpoint{
		PrefixHashes: append([]string(nil), checkpoint.PrefixHashes...),
		Replacement:  make([][]byte, 0, len(checkpoint.Replacement)),
	}
	for _, item := range checkpoint.Replacement {
		cloned.Replacement = append(cloned.Replacement, append([]byte(nil), item...))
	}
	return cloned
}

func codexCompactionCacheKey(modelName, sessionKey string) string {
	modelName = strings.TrimSpace(modelName)
	sessionKey = strings.TrimSpace(sessionKey)
	if modelName == "" || sessionKey == "" {
		return ""
	}
	// The session key is the continuity boundary. Keep this independent from
	// the selected upstream Codex credential so auth failover can preserve
	// compaction checkpoints.
	return strings.Join([]string{"codex-compaction", modelName, sessionKey}, "\x00")
}

func evictOldestCodexCompactionEntries(count int) {
	if count <= 0 || len(codexCompactionEntries) == 0 {
		return
	}
	type candidate struct {
		key       string
		timestamp time.Time
	}
	candidates := make([]candidate, 0, len(codexCompactionEntries))
	for key, entry := range codexCompactionEntries {
		candidates = append(candidates, candidate{key: key, timestamp: entry.Timestamp})
	}
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].timestamp.Before(candidates[j].timestamp)
	})
	if count > len(candidates) {
		count = len(candidates)
	}
	for i := 0; i < count; i++ {
		delete(codexCompactionEntries, candidates[i].key)
	}
}

func purgeExpiredCodexCompactionCache(now time.Time) {
	codexCompactionMu.Lock()
	for key, entry := range codexCompactionEntries {
		if now.Sub(entry.Timestamp) > CodexCompactionCacheTTL {
			delete(codexCompactionEntries, key)
		}
	}
	codexCompactionMu.Unlock()
}
