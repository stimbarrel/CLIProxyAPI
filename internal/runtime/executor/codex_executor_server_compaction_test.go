package executor

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	internalcache "github.com/router-for-me/CLIProxyAPI/v7/internal/cache"
	"github.com/router-for-me/CLIProxyAPI/v7/internal/config"
	cliproxyauth "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/auth"
	cliproxyexecutor "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/executor"
	sdktranslator "github.com/router-for-me/CLIProxyAPI/v7/sdk/translator"
	"github.com/tidwall/gjson"
)

const compactionTestEncryptedContent = "gAAAAABtestencryptedcompactionsummary"

func compactionTestClaudePayload(sessionID string, turns int) []byte {
	var messages []string
	for i := 0; i < turns; i++ {
		messages = append(messages,
			fmt.Sprintf(`{"role":"user","content":[{"type":"text","text":"user turn %d with some padding text to give the tokenizer something to count"}]}`, i),
			fmt.Sprintf(`{"role":"assistant","content":[{"type":"text","text":"assistant turn %d with matching padding text in the reply body"}]}`, i),
		)
	}
	return []byte(`{"model":"gpt-5.4","metadata":{"user_id":"{\"device_id\":\"device-test\",\"account_uuid\":\"\",\"session_id\":\"` + sessionID + `\"}"},"messages":[` + strings.Join(messages, ",") + `]}`)
}

func newCompactionTestServer(t *testing.T, compactCalls, responsesCalls *atomic.Int64, lastResponsesBody *atomic.Value) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		switch r.URL.Path {
		case "/responses/compact":
			compactCalls.Add(1)
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"id":"resp_compact","object":"response.compaction","output":[{"type":"message","role":"user","content":[{"type":"input_text","text":"retained user message"}]},{"type":"compaction_summary","encrypted_content":"` + compactionTestEncryptedContent + `"}],"usage":{"input_tokens":100,"output_tokens":10,"total_tokens":110}}`))
		case "/responses":
			responsesCalls.Add(1)
			if lastResponsesBody != nil {
				lastResponsesBody.Store(body)
			}
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = w.Write([]byte("data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"ok\"}]},\"output_index\":0}\n\n"))
			_, _ = w.Write([]byte("data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"status\":\"completed\",\"output\":[],\"usage\":{\"input_tokens\":8,\"output_tokens\":2,\"total_tokens\":10}}}\n\n"))
		default:
			t.Errorf("unexpected path %s", r.URL.Path)
			w.WriteHeader(http.StatusNotFound)
		}
	}))
}

func compactionTestAuth(baseURL string) *cliproxyauth.Auth {
	return &cliproxyauth.Auth{Attributes: map[string]string{
		"base_url": baseURL,
		"api_key":  "test",
	}}
}

func compactionTestConfig(triggerTokens int) *config.Config {
	cfg := &config.Config{}
	cfg.Codex.Compaction.TriggerTokens = triggerTokens
	return cfg
}

func TestCodexExecutorServerCompactionTriggersAndCachesCheckpoint(t *testing.T) {
	internalcache.ClearCodexCompactionCache()
	internalcache.ClearCodexReasoningReplayCache()

	var compactCalls, responsesCalls atomic.Int64
	var lastResponsesBody atomic.Value
	server := newCompactionTestServer(t, &compactCalls, &responsesCalls, &lastResponsesBody)
	defer server.Close()

	executor := NewCodexExecutor(compactionTestConfig(10))
	opts := cliproxyexecutor.Options{SourceFormat: sdktranslator.FromString("claude"), Stream: false}

	resp, err := executor.Execute(context.Background(), compactionTestAuth(server.URL), cliproxyexecutor.Request{
		Model:   "gpt-5.4",
		Payload: compactionTestClaudePayload("session-compact-1", 3),
	}, opts)
	if err != nil {
		t.Fatalf("Execute error: %v", err)
	}
	if len(resp.Payload) == 0 {
		t.Fatal("empty response payload")
	}
	if got := compactCalls.Load(); got != 1 {
		t.Fatalf("compact calls = %d, want 1", got)
	}
	if got := responsesCalls.Load(); got != 1 {
		t.Fatalf("responses calls = %d, want 1", got)
	}

	sent, _ := lastResponsesBody.Load().([]byte)
	input := gjson.GetBytes(sent, "input").Array()
	if len(input) != 2 {
		t.Fatalf("compacted input length = %d, want 2 (retained message + compaction item)", len(input))
	}
	if got := input[1].Get("type").String(); got != "compaction_summary" {
		t.Fatalf("last input item type = %q, want compaction_summary", got)
	}
	if got := input[1].Get("encrypted_content").String(); got != compactionTestEncryptedContent {
		t.Fatalf("encrypted_content = %q, want test blob", got)
	}
}

func TestCodexExecutorServerCompactionReusesCheckpointForGrownHistory(t *testing.T) {
	internalcache.ClearCodexCompactionCache()
	internalcache.ClearCodexReasoningReplayCache()

	var compactCalls, responsesCalls atomic.Int64
	var lastResponsesBody atomic.Value
	server := newCompactionTestServer(t, &compactCalls, &responsesCalls, &lastResponsesBody)
	defer server.Close()

	// High enough that only the first oversized request compacts; the second
	// request shrinks below the trigger once the checkpoint is substituted.
	executor := NewCodexExecutor(compactionTestConfig(60))
	opts := cliproxyexecutor.Options{SourceFormat: sdktranslator.FromString("claude"), Stream: false}
	auth := compactionTestAuth(server.URL)

	if _, err := executor.Execute(context.Background(), auth, cliproxyexecutor.Request{
		Model:   "gpt-5.4",
		Payload: compactionTestClaudePayload("session-compact-2", 4),
	}, opts); err != nil {
		t.Fatalf("first Execute error: %v", err)
	}
	if got := compactCalls.Load(); got != 1 {
		t.Fatalf("compact calls after first request = %d, want 1", got)
	}

	// Same history plus one more turn: the compacted prefix must be reused
	// without another compact call.
	if _, err := executor.Execute(context.Background(), auth, cliproxyexecutor.Request{
		Model:   "gpt-5.4",
		Payload: compactionTestClaudePayload("session-compact-2", 5),
	}, opts); err != nil {
		t.Fatalf("second Execute error: %v", err)
	}
	if got := compactCalls.Load(); got != 1 {
		t.Fatalf("compact calls after second request = %d, want 1 (checkpoint reuse)", got)
	}

	sent, _ := lastResponsesBody.Load().([]byte)
	input := gjson.GetBytes(sent, "input").Array()
	if len(input) < 3 {
		t.Fatalf("substituted input length = %d, want >= 3 (replacement + new suffix)", len(input))
	}
	if got := input[0].Get("content.0.text").String(); got != "retained user message" {
		t.Fatalf("first item text = %q, want retained user message", got)
	}
	if got := input[1].Get("type").String(); got != "compaction_summary" {
		t.Fatalf("second item type = %q, want compaction_summary", got)
	}
	last := input[len(input)-1]
	if !strings.Contains(last.Get("content.0.text").String(), "turn 4") {
		t.Fatalf("last item should be the new suffix turn, got %s", last.Raw)
	}
}

func TestCodexExecutorServerCompactionRetriesOnContextLengthError(t *testing.T) {
	internalcache.ClearCodexCompactionCache()
	internalcache.ClearCodexReasoningReplayCache()

	var compactCalls, responsesCalls atomic.Int64
	var lastResponsesBody atomic.Value
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		switch r.URL.Path {
		case "/responses/compact":
			compactCalls.Add(1)
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"output":[{"type":"message","role":"user","content":[{"type":"input_text","text":"retained user message"}]},{"type":"compaction_summary","encrypted_content":"` + compactionTestEncryptedContent + `"}]}`))
		case "/responses":
			responsesCalls.Add(1)
			if responsesCalls.Load() == 1 {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusBadRequest)
				_, _ = w.Write([]byte(`{"error":{"type":"invalid_request_error","code":"context_length_exceeded","message":"Your input exceeds the context window of this model."}}`))
				return
			}
			lastResponsesBody.Store(body)
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = w.Write([]byte("data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"status\":\"completed\",\"output\":[{\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"ok\"}]}],\"usage\":{\"input_tokens\":8,\"output_tokens\":2,\"total_tokens\":10}}}\n\n"))
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer server.Close()

	// Threshold too high to trigger proactively; the 400 must force the retry.
	executor := NewCodexExecutor(compactionTestConfig(1_000_000))
	opts := cliproxyexecutor.Options{SourceFormat: sdktranslator.FromString("claude"), Stream: false}

	resp, err := executor.Execute(context.Background(), compactionTestAuth(server.URL), cliproxyexecutor.Request{
		Model:   "gpt-5.4",
		Payload: compactionTestClaudePayload("session-compact-3", 2),
	}, opts)
	if err != nil {
		t.Fatalf("Execute error: %v", err)
	}
	if len(resp.Payload) == 0 {
		t.Fatal("empty response payload")
	}
	if got := compactCalls.Load(); got != 1 {
		t.Fatalf("compact calls = %d, want 1", got)
	}
	if got := responsesCalls.Load(); got != 2 {
		t.Fatalf("responses calls = %d, want 2 (fail then retry)", got)
	}
	sent, _ := lastResponsesBody.Load().([]byte)
	input := gjson.GetBytes(sent, "input").Array()
	if len(input) != 2 || input[1].Get("type").String() != "compaction_summary" {
		t.Fatalf("retried input not compacted: %s", gjson.GetBytes(sent, "input").Raw)
	}
}

func TestCodexExecutorServerCompactionDisabledByConfig(t *testing.T) {
	internalcache.ClearCodexCompactionCache()
	internalcache.ClearCodexReasoningReplayCache()

	var compactCalls, responsesCalls atomic.Int64
	server := newCompactionTestServer(t, &compactCalls, &responsesCalls, nil)
	defer server.Close()

	cfg := compactionTestConfig(10)
	cfg.Codex.Compaction.Disabled = true
	executor := NewCodexExecutor(cfg)
	opts := cliproxyexecutor.Options{SourceFormat: sdktranslator.FromString("claude"), Stream: false}

	if _, err := executor.Execute(context.Background(), compactionTestAuth(server.URL), cliproxyexecutor.Request{
		Model:   "gpt-5.4",
		Payload: compactionTestClaudePayload("session-compact-4", 3),
	}, opts); err != nil {
		t.Fatalf("Execute error: %v", err)
	}
	if got := compactCalls.Load(); got != 0 {
		t.Fatalf("compact calls = %d, want 0 when disabled", got)
	}
}

func TestCodexExecutorServerCompactionSkipsNativeResponsesClients(t *testing.T) {
	cfg := compactionTestConfig(10)
	if codexServerCompactionEnabled(cfg, sdktranslator.FromString("openai-response"), cliproxyexecutor.Options{}) {
		t.Fatal("native Responses clients must not trigger transparent compaction")
	}
	if !codexServerCompactionEnabled(cfg, sdktranslator.FromString("claude"), cliproxyexecutor.Options{}) {
		t.Fatal("claude source must be eligible for transparent compaction")
	}
	if codexServerCompactionEnabled(cfg, sdktranslator.FromString("claude"), cliproxyexecutor.Options{Alt: "responses/compact"}) {
		t.Fatal("explicit compact passthrough must not recurse")
	}
}

func TestCodexExecutorServerCompactionStreamPath(t *testing.T) {
	internalcache.ClearCodexCompactionCache()
	internalcache.ClearCodexReasoningReplayCache()

	var compactCalls, responsesCalls atomic.Int64
	var lastResponsesBody atomic.Value
	server := newCompactionTestServer(t, &compactCalls, &responsesCalls, &lastResponsesBody)
	defer server.Close()

	executor := NewCodexExecutor(compactionTestConfig(10))
	opts := cliproxyexecutor.Options{SourceFormat: sdktranslator.FromString("claude"), Stream: true}

	stream, err := executor.ExecuteStream(context.Background(), compactionTestAuth(server.URL), cliproxyexecutor.Request{
		Model:   "gpt-5.4",
		Payload: compactionTestClaudePayload("session-compact-5", 3),
	}, opts)
	if err != nil {
		t.Fatalf("ExecuteStream error: %v", err)
	}
	for chunk := range stream.Chunks {
		if chunk.Err != nil {
			t.Fatalf("stream chunk error: %v", chunk.Err)
		}
	}
	if got := compactCalls.Load(); got != 1 {
		t.Fatalf("compact calls = %d, want 1", got)
	}
	sent, _ := lastResponsesBody.Load().([]byte)
	input := gjson.GetBytes(sent, "input").Array()
	if len(input) != 2 || input[1].Get("type").String() != "compaction_summary" {
		t.Fatalf("stream request input not compacted: %s", gjson.GetBytes(sent, "input").Raw)
	}
}
