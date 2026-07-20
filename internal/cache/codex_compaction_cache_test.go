package cache

import (
	"fmt"
	"testing"
)

func compactionTestHashes(n int) []string {
	hashes := make([]string, 0, n)
	for i := 0; i < n; i++ {
		hashes = append(hashes, HashCodexCompactionItem([]byte(fmt.Sprintf(`{"item":%d}`, i))))
	}
	return hashes
}

func TestCodexCompactionCheckpointMatchLongestPrefixWins(t *testing.T) {
	ClearCodexCompactionCache()
	hashes := compactionTestHashes(20)

	CacheCodexCompactionCheckpoint("gpt-5.4", "session", CodexCompactionCheckpoint{
		PrefixHashes: hashes[:10],
		Replacement:  [][]byte{[]byte(`{"type":"compaction_summary","encrypted_content":"old"}`)},
	})
	CacheCodexCompactionCheckpoint("gpt-5.4", "session", CodexCompactionCheckpoint{
		PrefixHashes: hashes[:15],
		Replacement:  [][]byte{[]byte(`{"type":"compaction_summary","encrypted_content":"new"}`)},
	})

	checkpoint, ok := MatchCodexCompactionCheckpoint("gpt-5.4", "session", hashes)
	if !ok {
		t.Fatal("expected checkpoint match")
	}
	if len(checkpoint.PrefixHashes) != 15 {
		t.Fatalf("matched prefix length = %d, want 15 (newest checkpoint)", len(checkpoint.PrefixHashes))
	}
	if string(checkpoint.Replacement[0]) != `{"type":"compaction_summary","encrypted_content":"new"}` {
		t.Fatalf("unexpected replacement: %s", checkpoint.Replacement[0])
	}
}

func TestCodexCompactionCheckpointFallsBackToOlderPrefix(t *testing.T) {
	ClearCodexCompactionCache()
	hashes := compactionTestHashes(20)

	CacheCodexCompactionCheckpoint("gpt-5.4", "session", CodexCompactionCheckpoint{
		PrefixHashes: hashes[:10],
		Replacement:  [][]byte{[]byte(`{"a":1}`)},
	})
	// Newest checkpoint covers a prefix that diverges from the request below.
	diverged := append(append([]string(nil), hashes[:12]...), HashCodexCompactionItem([]byte(`{"edited":true}`)))
	CacheCodexCompactionCheckpoint("gpt-5.4", "session", CodexCompactionCheckpoint{
		PrefixHashes: diverged,
		Replacement:  [][]byte{[]byte(`{"b":2}`)},
	})

	checkpoint, ok := MatchCodexCompactionCheckpoint("gpt-5.4", "session", hashes)
	if !ok {
		t.Fatal("expected fallback match on older checkpoint")
	}
	if len(checkpoint.PrefixHashes) != 10 {
		t.Fatalf("matched prefix length = %d, want 10", len(checkpoint.PrefixHashes))
	}
}

func TestCodexCompactionCheckpointNoMatchOnDivergedHistory(t *testing.T) {
	ClearCodexCompactionCache()
	hashes := compactionTestHashes(10)

	CacheCodexCompactionCheckpoint("gpt-5.4", "session", CodexCompactionCheckpoint{
		PrefixHashes: hashes,
		Replacement:  [][]byte{[]byte(`{"a":1}`)},
	})

	shorter := hashes[:5]
	if _, ok := MatchCodexCompactionCheckpoint("gpt-5.4", "session", shorter); ok {
		t.Fatal("checkpoint longer than request input must not match")
	}
	if _, ok := MatchCodexCompactionCheckpoint("gpt-5.4", "other-session", hashes); ok {
		t.Fatal("different session must not match")
	}
	if _, ok := MatchCodexCompactionCheckpoint("gpt-5.5", "session", hashes); ok {
		t.Fatal("different model must not match")
	}
}

func TestCodexCompactionCheckpointCapsPerEntry(t *testing.T) {
	ClearCodexCompactionCache()
	hashes := compactionTestHashes(64)
	for i := 1; i <= CodexCompactionMaxCheckpointsPerEntry+4; i++ {
		CacheCodexCompactionCheckpoint("gpt-5.4", "session", CodexCompactionCheckpoint{
			PrefixHashes: hashes[:i],
			Replacement:  [][]byte{[]byte(`{"n":1}`)},
		})
	}
	codexCompactionMu.Lock()
	entry := codexCompactionEntries[codexCompactionCacheKey("gpt-5.4", "session")]
	codexCompactionMu.Unlock()
	if len(entry.Checkpoints) != CodexCompactionMaxCheckpointsPerEntry {
		t.Fatalf("checkpoints = %d, want %d", len(entry.Checkpoints), CodexCompactionMaxCheckpointsPerEntry)
	}
}
