// Regression test for the qwen4exp MTP shared output head.
//
// Some quantized qwen4exp GGUFs retain `blk.N.nextn.shared_head_head.weight`
// (a copy of the LM head kept by the reference MTP module) even though the
// llama.cpp qwen4exp implementation computes all logits from the trunk head.
// The loader used to fail with
//   done_getting_tensors: wrong number of tensors; expected %d, got %d
// because that tensor was never requested. The fix requests it with
// TENSOR_NOT_REQUIRED | TENSOR_SKIP, so it is consumed but not loaded.
//
// This test synthesizes a minimal file-backed qwen4exp GGUF containing every
// tensor the loader expects, including the shared head, and asserts that the
// model loads. It must go through the real file-load path: only there does
// done_getting_tensors() reconcile the model's tensor set against the file's
// inventory (llama_model_init_from_user() creates missing tensors virtually
// and never exercises the count check).

#include "ggml.h"
#include "ggml-cpp.h"
#include "gguf.h"
#include "llama.h"

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// tiny but structurally complete: 8 trunk layers (3 linear + 1 full attention
// per 4, like the real arch) plus 1 NextN block
constexpr uint32_t N_VOCAB       = 128;
constexpr uint32_t N_EMBD        = 64;
constexpr uint32_t N_HEAD        = 4;
constexpr uint32_t N_HEAD_KV     = 2;
constexpr uint32_t HEAD_DIM      = 16;
constexpr uint32_t N_LAYER       = 8;  // trunk
constexpr uint32_t N_NEXTN       = 1;
constexpr uint32_t N_LAYER_ALL   = N_LAYER + N_NEXTN; // = block_count
constexpr uint32_t N_EXPERT      = 4;
constexpr uint32_t N_USED        = 2;
constexpr uint32_t N_FFN_EXP     = 32;
constexpr uint32_t HC            = 2;   // hyper-connection streams
constexpr uint32_t HC_DIM        = HC * N_EMBD;
constexpr uint32_t HC_LR         = 8;   // hyper-connection low rank
constexpr uint32_t D_STATE       = 8;   // ssm.state_size == head dim (k and v)
constexpr uint32_t N_GROUP       = 2;   // ssm group count == k heads
constexpr uint32_t DT_RANK       = 4;   // ssm time step rank == v heads
constexpr uint32_t CONV_KERNEL   = 4;
// linear-attention projections, per qwen4exp.cpp load_block():
// key = d_state*n_group, value = d_state*dt_rank, conv = 2*key + value
constexpr uint32_t SSM_KEY_DIM   = D_STATE * N_GROUP;
constexpr uint32_t SSM_VAL_DIM   = D_STATE * DT_RANK;
constexpr uint32_t SSM_CONV_DIM  = SSM_KEY_DIM * 2 + SSM_VAL_DIM;
constexpr uint32_t N_EMBD_K_GQA  = HEAD_DIM * N_HEAD_KV;
constexpr uint32_t N_EMBD_V_GQA  = HEAD_DIM * N_HEAD_KV;
constexpr uint32_t N_EMBD_Q      = HEAD_DIM * N_HEAD * 2; // q carries a per-head gate
constexpr uint32_t IDX_DIM       = 16;                    // indexer key length
constexpr uint32_t IDX_HEADS     = 2;

void set_kv_u32_arr(gguf_context * gguf, const char * key, const std::vector<uint32_t> & v) {
    gguf_set_arr_data(gguf, key, GGUF_TYPE_UINT32, v.data(), v.size());
}

void add_tensor(gguf_context * gguf, ggml_context * ctx, const char * name,
                std::initializer_list<int64_t> ne) {
    ggml_tensor * t = ggml_new_tensor(ctx, GGML_TYPE_F32, ne.size(), ne.begin());
    ggml_set_name(t, name);
    ggml_set_zero(t);
    gguf_add_tensor(gguf, t);
}

bool is_full_attention(uint32_t il) {
    // the loader derives the trunk pattern from full_attention_interval = 4;
    // the NextN block is always full attention
    return il >= N_LAYER || (il + 1) % 4 == 0;
}

void add_block_tensors(gguf_context * gguf, ggml_context * ctx, uint32_t il) {
    char name[128];

    // hyper-connection mixers, before the token mixer and before the MoE
    auto add_hc = [&](const char * what) {
        snprintf(name, sizeof(name), "blk.%" PRIu32 ".hc_%s_norm.weight",   il, what);
        add_tensor(gguf, ctx, name, { N_EMBD, HC });
        snprintf(name, sizeof(name), "blk.%" PRIu32 ".hc_%s_down.weight",   il, what);
        add_tensor(gguf, ctx, name, { HC_DIM, HC_LR });
        snprintf(name, sizeof(name), "blk.%" PRIu32 ".hc_%s_up.weight",     il, what);
        add_tensor(gguf, ctx, name, { HC_LR, HC_DIM });
        snprintf(name, sizeof(name), "blk.%" PRIu32 ".hc_%s_inject.weight", il, what);
        add_tensor(gguf, ctx, name, { HC_DIM, HC });
    };
    add_hc("attn");
    add_hc("ffn");

    if (is_full_attention(il)) {
        snprintf(name, sizeof(name), "blk.%" PRIu32 ".attn_q.weight", il);
        add_tensor(gguf, ctx, name, { N_EMBD, N_EMBD_Q });
        snprintf(name, sizeof(name), "blk.%" PRIu32 ".attn_k.weight", il);
        add_tensor(gguf, ctx, name, { N_EMBD, N_EMBD_K_GQA });
        snprintf(name, sizeof(name), "blk.%" PRIu32 ".attn_v.weight", il);
        add_tensor(gguf, ctx, name, { N_EMBD, N_EMBD_V_GQA });
        snprintf(name, sizeof(name), "blk.%" PRIu32 ".attn_output.weight", il);
        add_tensor(gguf, ctx, name, { HEAD_DIM * N_HEAD, N_EMBD });
        snprintf(name, sizeof(name), "blk.%" PRIu32 ".attn_q_norm.weight", il);
        add_tensor(gguf, ctx, name, { HEAD_DIM });
        snprintf(name, sizeof(name), "blk.%" PRIu32 ".attn_k_norm.weight", il);
        add_tensor(gguf, ctx, name, { HEAD_DIM });
        snprintf(name, sizeof(name), "blk.%" PRIu32 ".indexer.q_proj.weight", il);
        add_tensor(gguf, ctx, name, { N_EMBD, IDX_HEADS * IDX_DIM });
        snprintf(name, sizeof(name), "blk.%" PRIu32 ".indexer.k_proj.weight", il);
        add_tensor(gguf, ctx, name, { N_EMBD, IDX_DIM });
        snprintf(name, sizeof(name), "blk.%" PRIu32 ".indexer.q_norm.weight", il);
        add_tensor(gguf, ctx, name, { IDX_DIM });
        snprintf(name, sizeof(name), "blk.%" PRIu32 ".indexer.k_norm.weight", il);
        add_tensor(gguf, ctx, name, { IDX_DIM });
    } else {
        snprintf(name, sizeof(name), "blk.%" PRIu32 ".attn_qkv.weight", il);
        add_tensor(gguf, ctx, name, { N_EMBD, SSM_KEY_DIM * 2 + SSM_VAL_DIM });
        snprintf(name, sizeof(name), "blk.%" PRIu32 ".attn_gate.weight", il);
        add_tensor(gguf, ctx, name, { N_EMBD, SSM_VAL_DIM });
        snprintf(name, sizeof(name), "blk.%" PRIu32 ".ssm_conv1d.weight", il);
        add_tensor(gguf, ctx, name, { CONV_KERNEL, SSM_CONV_DIM });
        snprintf(name, sizeof(name), "blk.%" PRIu32 ".ssm_dt.bias", il);
        add_tensor(gguf, ctx, name, { DT_RANK });
        snprintf(name, sizeof(name), "blk.%" PRIu32 ".ssm_a", il);
        add_tensor(gguf, ctx, name, { DT_RANK });
        snprintf(name, sizeof(name), "blk.%" PRIu32 ".ssm_beta.weight", il);
        add_tensor(gguf, ctx, name, { N_EMBD, DT_RANK });
        snprintf(name, sizeof(name), "blk.%" PRIu32 ".ssm_alpha.weight", il);
        add_tensor(gguf, ctx, name, { N_EMBD, DT_RANK });
        snprintf(name, sizeof(name), "blk.%" PRIu32 ".ssm_norm.weight", il);
        add_tensor(gguf, ctx, name, { D_STATE });
        snprintf(name, sizeof(name), "blk.%" PRIu32 ".ssm_out.weight", il);
        add_tensor(gguf, ctx, name, { SSM_VAL_DIM, N_EMBD });
    }

    snprintf(name, sizeof(name), "blk.%" PRIu32 ".ffn_gate_inp.weight", il);
    add_tensor(gguf, ctx, name, { N_EMBD, N_EXPERT });
    snprintf(name, sizeof(name), "blk.%" PRIu32 ".ffn_down_exps.weight", il);
    add_tensor(gguf, ctx, name, { N_FFN_EXP, N_EMBD, N_EXPERT });
    snprintf(name, sizeof(name), "blk.%" PRIu32 ".ffn_gate_exps.weight", il);
    add_tensor(gguf, ctx, name, { N_EMBD, N_FFN_EXP, N_EXPERT });
    snprintf(name, sizeof(name), "blk.%" PRIu32 ".ffn_up_exps.weight", il);
    add_tensor(gguf, ctx, name, { N_EMBD, N_FFN_EXP, N_EXPERT });
    snprintf(name, sizeof(name), "blk.%" PRIu32 ".ffn_gate_inp_shexp.weight", il);
    add_tensor(gguf, ctx, name, { N_EMBD });
    snprintf(name, sizeof(name), "blk.%" PRIu32 ".ffn_gate_shexp.weight", il);
    add_tensor(gguf, ctx, name, { N_EMBD, N_FFN_EXP });
    snprintf(name, sizeof(name), "blk.%" PRIu32 ".ffn_up_shexp.weight", il);
    add_tensor(gguf, ctx, name, { N_EMBD, N_FFN_EXP });
    snprintf(name, sizeof(name), "blk.%" PRIu32 ".ffn_down_shexp.weight", il);
    add_tensor(gguf, ctx, name, { N_FFN_EXP, N_EMBD });
}

void write_test_gguf(const std::string & path) {
    gguf_context_ptr gguf(gguf_init_empty());

    gguf_set_val_str(gguf.get(), "general.architecture", "qwen4exp");

    gguf_set_val_u32(gguf.get(), "qwen4exp.vocab_size",       N_VOCAB);
    gguf_set_val_u32(gguf.get(), "qwen4exp.context_length",   256);
    gguf_set_val_u32(gguf.get(), "qwen4exp.embedding_length", N_EMBD);
    gguf_set_val_u32(gguf.get(), "qwen4exp.block_count",      N_LAYER_ALL);
    gguf_set_val_u32(gguf.get(), "qwen4exp.nextn_predict_layers", N_NEXTN);

    gguf_set_val_u32(gguf.get(), "qwen4exp.attention.head_count",    N_HEAD);
    gguf_set_val_u32(gguf.get(), "qwen4exp.attention.head_count_kv", N_HEAD_KV);
    gguf_set_val_u32(gguf.get(), "qwen4exp.attention.key_length",    HEAD_DIM);
    gguf_set_val_u32(gguf.get(), "qwen4exp.attention.value_length",  HEAD_DIM);
    gguf_set_val_f32(gguf.get(), "qwen4exp.attention.layer_norm_rms_epsilon", 1e-5f);

    gguf_set_val_u32(gguf.get(), "qwen4exp.rope.dimension_count", HEAD_DIM);
    set_kv_u32_arr(gguf.get(), "qwen4exp.rope.dimension_sections", { 4, 4, 4, 4 });
    gguf_set_val_f32(gguf.get(), "qwen4exp.rope.freq_base", 10000.0f);

    gguf_set_val_u32(gguf.get(), "qwen4exp.expert_count",                      N_EXPERT);
    gguf_set_val_u32(gguf.get(), "qwen4exp.expert_used_count",                 N_USED);
    gguf_set_val_u32(gguf.get(), "qwen4exp.expert_feed_forward_length",        N_FFN_EXP);
    gguf_set_val_u32(gguf.get(), "qwen4exp.expert_shared_feed_forward_length", N_FFN_EXP);

    gguf_set_val_u32(gguf.get(), "qwen4exp.ssm.conv_kernel",    CONV_KERNEL);
    gguf_set_val_u32(gguf.get(), "qwen4exp.ssm.state_size",     D_STATE);
    gguf_set_val_u32(gguf.get(), "qwen4exp.ssm.group_count",    N_GROUP);
    gguf_set_val_u32(gguf.get(), "qwen4exp.ssm.time_step_rank", DT_RANK);
    gguf_set_val_u32(gguf.get(), "qwen4exp.ssm.inner_size",     SSM_VAL_DIM);

    gguf_set_val_u32(gguf.get(), "qwen4exp.full_attention_interval", 4);
    gguf_set_val_u32(gguf.get(), "qwen4exp.hyper_connection.count",    HC);
    gguf_set_val_u32(gguf.get(), "qwen4exp.hyper_connection.low_rank", HC_LR);
    gguf_set_val_u32(gguf.get(), "qwen4exp.attention.indexer.head_count", IDX_HEADS);
    gguf_set_val_u32(gguf.get(), "qwen4exp.attention.indexer.key_length", IDX_DIM);
    gguf_set_val_u32(gguf.get(), "qwen4exp.attention.indexer.top_k",      8);

    // dummy vocab, same scheme as test-llama-archs
    gguf_set_val_str(gguf.get(), "tokenizer.ggml.model", "test");
    {
        std::vector<std::string> tokens;
        tokens.reserve(N_VOCAB);
        for (uint32_t i = 0; i < N_VOCAB; i++) {
            tokens.push_back("tok_" + std::to_string(i));
        }
        std::vector<const char *> token_ptrs;
        token_ptrs.reserve(tokens.size());
        for (const auto & t : tokens) {
            token_ptrs.push_back(t.c_str());
        }
        std::vector<float> scores(N_VOCAB, 0.0f);
        gguf_set_arr_str(gguf.get(), "tokenizer.ggml.tokens", token_ptrs.data(), token_ptrs.size());
        gguf_set_arr_data(gguf.get(), "tokenizer.ggml.scores", GGUF_TYPE_FLOAT32, scores.data(), scores.size());
    }

    // the ggml context owns the tensor data until the file is written;
    // all tensors are tiny, a few MiB cover them with room to spare
    ggml_context_ptr ctx(ggml_init({ 8*1024*1024, NULL, /*no_alloc*/ false }));

    add_tensor(gguf.get(), ctx.get(), "token_embd.weight",     { N_EMBD, N_VOCAB });
    add_tensor(gguf.get(), ctx.get(), "output.weight",         { N_EMBD, N_VOCAB });
    add_tensor(gguf.get(), ctx.get(), "output_hc_norm.weight", { N_EMBD, HC });
    add_tensor(gguf.get(), ctx.get(), "output_hc_down.weight", { HC_DIM, HC_LR });
    add_tensor(gguf.get(), ctx.get(), "output_hc_up.weight",   { HC_LR, HC_DIM });

    for (uint32_t il = 0; il < N_LAYER_ALL; il++) {
        add_block_tensors(gguf.get(), ctx.get(), il);
    }

    // the NextN block's own tensors
    add_tensor(gguf.get(), ctx.get(), "blk.8.nextn.eh_proj.weight",      { 2 * N_EMBD, N_EMBD });
    add_tensor(gguf.get(), ctx.get(), "blk.8.nextn.enorm.weight",        { N_EMBD });
    add_tensor(gguf.get(), ctx.get(), "blk.8.nextn.hnorm.weight",        { N_EMBD, HC });
    add_tensor(gguf.get(), ctx.get(), "blk.8.nextn.hc_head_norm.weight", { N_EMBD, HC });
    add_tensor(gguf.get(), ctx.get(), "blk.8.nextn.hc_head_down.weight", { HC_DIM, HC_LR });
    add_tensor(gguf.get(), ctx.get(), "blk.8.nextn.hc_head_up.weight",   { HC_LR, HC_DIM });

    // the tensor under test: a copy of the LM head retained by the reference
    // MTP module, which the llama.cpp loader must tolerate
    add_tensor(gguf.get(), ctx.get(), "blk.8.nextn.shared_head_head.weight", { N_EMBD, N_VOCAB });

    if (!gguf_write_to_file(gguf.get(), path.c_str(), /*only_meta*/ false)) {
        throw std::runtime_error("failed to write " + path);
    }
}

int run() {
    const auto path = std::filesystem::temp_directory_path() / "test-qwen4exp-mtp-shared-head.gguf";
    write_test_gguf(path.string());

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;

    llama_model * model = llama_model_load_from_file(path.string().c_str(), mparams);

    if (!model) {
        fprintf(stderr, "%s: model load failed (see the llama.cpp log above)\n", __func__);
        std::filesystem::remove(path);
        return 1;
    }

    // light sanity check that a real, usable model came out
    const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    llama_model_free(model);
    std::filesystem::remove(path);

    if (n_vocab != N_VOCAB) {
        fprintf(stderr, "%s: unexpected vocab size %u, expected %u\n", __func__, n_vocab, N_VOCAB);
        return 1;
    }
    return 0;
}

} // namespace

int main() {
    try {
        return run();
    } catch (const std::exception & err) {
        fprintf(stderr, "main: error: %s\n", err.what());
        return 1;
    }
}
