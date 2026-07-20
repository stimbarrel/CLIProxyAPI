package helps

import (
	"strings"

	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

// CodexCompactionItemTypes lists the Responses item types that carry an
// upstream compaction summary. The backend historically emits
// "compaction_summary" while newer Codex clients use "compaction".
var codexCompactionItemTypes = map[string]bool{
	"compaction":         true,
	"compaction_summary": true,
}

// IsCodexCompactionItemType reports whether the item type carries an upstream
// compaction summary.
func IsCodexCompactionItemType(itemType string) bool {
	return codexCompactionItemTypes[strings.TrimSpace(itemType)]
}

// CodexInputItemsRaw returns the raw JSON of each translated input item.
func CodexInputItemsRaw(body []byte) [][]byte {
	items := gjson.GetBytes(body, "input")
	if !items.IsArray() {
		return nil
	}
	arr := items.Array()
	raw := make([][]byte, 0, len(arr))
	for i := range arr {
		raw = append(raw, []byte(arr[i].Raw))
	}
	return raw
}

// codexCompactPayloadFields mirrors the Codex CLI ApiCompactionInput struct:
// /responses/compact rejects any parameter outside this set.
var codexCompactPayloadFields = []string{
	"model",
	"input",
	"instructions",
	"tools",
	"parallel_tool_calls",
	"reasoning",
	"service_tier",
	"prompt_cache_key",
	"text",
}

// BuildCodexCompactPayload derives the unary /responses/compact payload from a
// translated /responses request body, keeping only the fields the compact
// endpoint accepts.
func BuildCodexCompactPayload(body []byte) []byte {
	payload := []byte(`{}`)
	for _, field := range codexCompactPayloadFields {
		value := gjson.GetBytes(body, field)
		if !value.Exists() {
			continue
		}
		payload, _ = sjson.SetRawBytes(payload, field, []byte(value.Raw))
	}
	return payload
}

// ReplaceCodexInputItems swaps the entire input array of a translated request
// body with the provided replacement items.
func ReplaceCodexInputItems(body []byte, replacement [][]byte) ([]byte, bool) {
	updated, err := sjson.SetRawBytes(body, "input", []byte("[]"))
	if err != nil {
		return body, false
	}
	for _, item := range replacement {
		next, errSet := sjson.SetRawBytes(updated, "input.-1", item)
		if errSet != nil {
			return body, false
		}
		updated = next
	}
	return updated, true
}

// ReplaceCodexInputPrefix substitutes the first coveredCount input items with
// the replacement history while keeping the remaining suffix items.
func ReplaceCodexInputPrefix(body []byte, replacement [][]byte, coveredCount int) ([]byte, bool) {
	items := CodexInputItemsRaw(body)
	if coveredCount <= 0 || coveredCount > len(items) || len(replacement) == 0 {
		return body, false
	}
	merged := make([][]byte, 0, len(replacement)+len(items)-coveredCount)
	merged = append(merged, replacement...)
	merged = append(merged, items[coveredCount:]...)
	return ReplaceCodexInputItems(body, merged)
}

// ParseCodexCompactOutput extracts the replacement history from a unary
// /responses/compact response. It requires exactly one compaction item so a
// malformed upstream response never silently truncates history.
func ParseCodexCompactOutput(payload []byte) ([][]byte, bool) {
	output := gjson.GetBytes(payload, "output")
	if !output.IsArray() {
		return nil, false
	}
	arr := output.Array()
	items := make([][]byte, 0, len(arr))
	compactionCount := 0
	for i := range arr {
		if IsCodexCompactionItemType(arr[i].Get("type").String()) {
			compactionCount++
		}
		items = append(items, []byte(arr[i].Raw))
	}
	if compactionCount != 1 || len(items) == 0 {
		return nil, false
	}
	return items, true
}

// EstimateCodexEncryptedContentTokens approximates the token weight of opaque
// encrypted reasoning/compaction payloads that plain text token counting
// cannot see. The encrypted blob is roughly proportional to the hidden token
// count; one token per four bytes keeps the estimate conservative.
func EstimateCodexEncryptedContentTokens(body []byte) int64 {
	items := gjson.GetBytes(body, "input")
	if !items.IsArray() {
		return 0
	}
	var total int64
	arr := items.Array()
	for i := range arr {
		itemType := arr[i].Get("type").String()
		if itemType != "reasoning" && !IsCodexCompactionItemType(itemType) {
			continue
		}
		total += int64(len(arr[i].Get("encrypted_content").String()) / 4)
	}
	return total
}
