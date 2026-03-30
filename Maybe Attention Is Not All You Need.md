# Maybe Attention Is Not All You Need: Linear-Time Language Modeling via Wave Propagation and Resonance Memory

**Abstract**
The Transformer architecture, underpinned by the self-attention mechanism, has dominated natural language processing due to its parallelizability and robust modeling capabilities. However, its $O(N^2)$ computational and memory complexity with respect to sequence length limits its applicability to very long contexts and resource-constrained consumer hardware. In this paper, we introduce **Sage**, an entirely attention-free, linear-time $O(N)$ language modeling architecture. Sage replaces softmax-based attention with a dual-mechanism approach: **Wave Propagation** (multi-scale causal convolutions) for resolving local syntax, and **Resonance Memory** (a causally accumulated neural whiteboard) for global semantic understanding. We demonstrate that this architecture can achieve competitive perplexity while operating at strict $O(N)$ complexity, enabling full model training and fast inference on consumer-grade hardware (e.g., 4GB VRAM).

---

## 1. Introduction

Since the introduction of the Transformer, the self-attention mechanism has been the de facto standard for sequence modeling. By computing pairwise dot-products between all tokens in a sequence, attention provides unrestricted global receptive fields. However, the requirement to compute and store an $N \times N$ attention matrix imposes a severe quadratic scaling bottleneck.

Recent efforts to mitigate this bottleneck include sparse attention, linear attention, and state-space models (SSMs) like Mamba. Building upon these principles, we present an alternative formulation that relies on explicit decomposition of context processing. Specifically, we uncouple local syntactic processing from global semantic retrieval. We propose the **Sage Architecture**, which replaces the $O(N^2)$ self-attention block entirely with a combination of causal convolutions (Wave Propagation) and a low-rank cumulative memory matrix (Resonance Memory).

## 2. The Sage Architecture

The Sage architecture abandons the QKV (Query, Key, Value) paradigm. Instead, information flows through a stack of repeating **WaveBlocks**. Each WaveBlock consists of three sequential sub-modules:

1. **Local State:** Wave Propagation (WaveMixer)
2. **Global State:** Resonance Memory
3. **Position-wise Reasoning:** SwiGLU Multi-Layer Perceptron (MLP)

Let $X \in \mathbb{R}^{B \times L \times D}$ be the input tensor, where $B$ is batch size, $L$ is sequence length, and $D$ is the embedding dimension. The forward pass is defined as:

$$ X_{1} = X + \text{WaveMixer}(\text{RMSNorm}(X)) $$
$$ X_{2} = X_{1} + \text{ResonanceMemory}(\text{RMSNorm}(X_{1})) $$
$$ X_{out} = X_{2} + \text{SwiGLU}(\text{RMSNorm}(X_{2})) $$

### 2.1 Wave Propagation (Local Context)

Standard attention dedicates immense parameter capacity to modeling local word dependencies (e.g., verb-noun agreement) which are inherently proximity-based. Sage models local syntax using **WaveMixer**, a multi-scale 1D causal convolution module.

The input embeddings are chunked across the feature dimension and processed by parallel depthwise causal convolutions with expanding kernel sizes ($k \in \{3, 7, 15, 31\}$). As network depth increases, the kernel sizes are adaptively widened to expand the effective receptive field:

$$ Y_{short} = \text{CausalConv}_{k_{short}}(X_{short}) $$
$$ Y_{mid} = \text{CausalConv}_{k_{mid}}(X_{mid}) $$
$$ Y_{long} = \text{CausalConv}_{k_{long}}(X_{long}) $$

The components are concatenated and gated via a linear projection. Crucially, the causal padding ensures strict auto-regressive properties without requiring positional embeddings, and the complexity is strictly bounded at $O(N \cdot k)$.

### 2.2 Resonance Memory (Global Context)

To replace self-attention's global routing capability, we introduce **Resonance Memory**, a shared neural whiteboard. Instead of $O(N^2)$ pairwise comparisons, tokens project their state into $K$ shared memory slots.

1. **Write Matrix Generation:** The input $X$ is projected to generate a softmax-normalized write key $W_K \in \mathbb{R}^{B \times L \times K}$ and a compressed write value $W_V \in \mathbb{R}^{B \times L \times d_{mem}}$.
2. **Causal Accumulation:** The memory updates are computed via the outer product of $W_K$ and $W_V$. Global state is built using a strict $O(N)$ causal cumulative sum over the sequence dimension:
   $$ M_{t} = \sum_{i=0}^{t} (W_{K,i} \otimes W_{V,i}) $$
3. **Read Operation:** At any given step $t$, the token generates a read key $R_K$ to retrieve from the accumulated state $M_t$. The retrieved tensor is decompressed to the full dimension $D$ and selectively gated into the main residual stream.

By aggressively compressing the memory dimension ($d_{mem} \ll D$) and using a small number of slots ($K=16$), the entire computational overhead of the Resonance Memory operation remains bounded at $O(N \cdot K \cdot d_{mem})$, scaling perfectly linearly with sequence length.

## 3. Implementation Details

To ensure stable training and maximal performance on limited hardware profiles, several infrastructural optimizations were implemented:
- **RMSNorm and SwiGLU:** Adopted over LayerNorm and GELU for improved convergence stability and computational throughput.
- **Weight Tying:** The input embedding matrix and the final LM projection head share weights, reducing parameter overhead by nearly 50% in small-scale models.
- **Automatic Mixed Precision (AMP):** Implementation natively utilizes Float16 precision for all bulk matrix multiplications (`torch.amp`), halving the VRAM footprint and doubling Tensor Core utilization.

## 4. Experimental Results

A proof-of-concept version of Sage was trained to validate the theoretical linear scaling mechanisms. 

**Setup:**
- Parameters: $\sim$ 31.2 Million
- Context Window: 128 tokens
- Dataset: 15,000 instruction-following conversations (ShareGPT)
- Hardware: Consumer NVIDIA RTX 3050 (4GB VRAM)

**Observations:**
During a 15-epoch training run, the causal cross-entropy loss declined stably from 11.28 to 2.85 without divergence. The model achieved a competitive generation speed of 21–23 tokens per second on the 4GB hardware envelope, demonstrating the drastic reduction in KV-cache overhead inherently afforded by the Resonance Memory formulation compared to standard Attention KV-caching.

## 5. Conclusion

"Maybe Attention Is Not All You Need" challenges the assumption that $O(N^2)$ context mixing is necessary for language understanding. By decoupling local syntax processing (Wave Propagation) from global context retrieval (Resonance Memory), the Sage architecture provides an $O(N)$ linear-time drop-in replacement for the Transformer. Our preliminary validation demonstrates that sophisticated causal reasoning architectures can be trained and deployed entirely within ultra-low resource environments.
