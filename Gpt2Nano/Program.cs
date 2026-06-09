// =============================================================================
// GPT-2 Nano — .NET inference PoC using TorchSharp
//
// Loads a trained GPT-2 nano model (44M params, 12 layers, 512-dim) from
// SafeTensors format and generates text with streaming output.
//
// Features demonstrated:
//   - SafeTensors loading with full validation (no pickle, no code execution)
//   - Scaled dot-product attention (SDPA) for efficient attention computation
//   - Temperature / top-k / top-p sampling
//   - IAsyncEnumerable streaming token generation
//   - DisposeScope-based tensor memory management
//
// Usage:
//   dotnet run [--model <path>] [--tokenizer <path>] [--prompt "text"]
//
// Model source:
//   Train: https://github.com/kotlarmilos/gpt2-nano
//   Download: https://huggingface.co/kotlarmilos/gpt2-nano
// =============================================================================

using System.Runtime.CompilerServices;
using System.Text.Json;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

// ---- Parse arguments ----

var modelPath = GetArg(args, "--model") ?? FindDefault("model.safetensors", "weights");
var tokenizerDir = GetArg(args, "--tokenizer") ?? FindDefault("vocab.json", "data/bpe-tokenizer", returnDir: true);
var userPrompt = GetArg(args, "--prompt") ?? "The meaning of life is";
var maxTokens = int.TryParse(GetArg(args, "--tokens"), out var t) ? t : 60;

if (modelPath is null || !File.Exists(modelPath))
{
    Console.Error.WriteLine("Error: Model weights not found.");
    Console.Error.WriteLine();
    Console.Error.WriteLine("Provide a path to model.safetensors via --model, or place it in ./weights/");
    Console.Error.WriteLine();
    Console.Error.WriteLine("To get the weights:");
    Console.Error.WriteLine("  Option 1: Train from scratch — https://github.com/kotlarmilos/gpt2-nano");
    Console.Error.WriteLine("  Option 2: Download from HuggingFace — https://huggingface.co/kotlarmilos/gpt2-nano");
    Console.Error.WriteLine();
    Console.Error.WriteLine("Then export to SafeTensors:");
    Console.Error.WriteLine("  python -c \"import torch; from safetensors.torch import save_file; " +
        "cp = torch.load('checkpoints/final.pt', map_location='cpu', weights_only=False); " +
        "save_file({k: v.float() for k, v in cp['model_state_dict'].items()}, 'weights/model.safetensors')\"");
    return 1;
}

if (tokenizerDir is null || !File.Exists(Path.Combine(tokenizerDir, "vocab.json")))
{
    Console.Error.WriteLine("Error: Tokenizer files not found.");
    Console.Error.WriteLine();
    Console.Error.WriteLine("Provide a path to the tokenizer directory via --tokenizer,");
    Console.Error.WriteLine("or place vocab.json and merges.json in ./data/bpe-tokenizer/");
    return 1;
}

// ---- Model configuration (must match training) ----

const int ContextLen = 1024;
const int EmbeddingDim = 512;
const int NumHeads = 8;
const int NumLayers = 12;

// ---- Load tokenizer ----

var (merges, vocab) = LoadTokenizer(tokenizerDir);
var inverseVocab = vocab.ToDictionary(kv => kv.Value, kv => kv.Key);

// ---- Load model ----

var model = new GPT(vocab.Count, ContextLen, EmbeddingDim, NumLayers, NumHeads);
try
{
    LoadSafeTensors(model, modelPath);
}
catch (Exception ex) when (ex is InvalidDataException or NotSupportedException)
{
    Console.Error.WriteLine(ex.Message);
    return 1;
}
model.eval();

var paramCount = model.parameters().Sum(p => p.numel());

// ---- Interactive demo ----

var sampling = new SamplingConfig(Temperature: 0.8f, TopK: 50, TopP: 0.9f);

Console.ForegroundColor = ConsoleColor.Cyan;
Console.WriteLine("┌─────────────────────────────────────────────────────────────────┐");
Console.WriteLine("│  GPT-2 Nano — Pure .NET Inference                              │");
Console.WriteLine("│  Train in Python, infer in C#. No Python runtime needed.        │");
Console.WriteLine("└─────────────────────────────────────────────────────────────────┘");
Console.ResetColor();
Console.WriteLine();
Console.ForegroundColor = ConsoleColor.DarkGray;
Console.WriteLine($"  Model:      {paramCount:#,0} parameters (12 layers, 512-dim, 8 heads)");
Console.WriteLine($"  Weights:    {Path.GetFileName(modelPath)} (SafeTensors — no pickle, no code execution)");
Console.WriteLine($"  Attention:  scaled_dot_product_attention (SDPA)");
Console.WriteLine($"  Tokenizer:  BPE ({vocab.Count:#,0} tokens)");
Console.WriteLine($"  Sampling:   temp={sampling.Temperature}, top-k={sampling.TopK}, top-p={sampling.TopP}");
Console.WriteLine($"  Engine:     TorchSharp on .NET {Environment.Version}");
Console.ResetColor();
Console.WriteLine();
Console.ForegroundColor = ConsoleColor.DarkGray;
Console.WriteLine("  Type a prompt and press Enter. The model completes your text.");
Console.WriteLine("  Commands: /tokens <n>  /temp <f>  /topk <n>  /topp <f>  /quit");
Console.ResetColor();
Console.WriteLine();

// First auto-prompt
var prompt = userPrompt;
var firstRun = true;

while (true)
{
    if (!firstRun)
    {
        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.Write("prompt» ");
        Console.ResetColor();

        var line = Console.ReadLine();
        if (line is null) break;
        line = line.Trim();
        if (line.Length == 0) continue;

        if (line.StartsWith("/quit") || line.StartsWith("/exit"))
            break;
        if (line.StartsWith("/tokens ") && int.TryParse(line[8..], out var newMax))
        {
            maxTokens = newMax;
            Console.ForegroundColor = ConsoleColor.DarkGray;
            Console.WriteLine($"  → max tokens: {maxTokens}");
            Console.ResetColor();
            continue;
        }
        if (line.StartsWith("/temp ") && float.TryParse(line[6..], out var newTemp))
        {
            sampling = sampling with { Temperature = newTemp };
            Console.ForegroundColor = ConsoleColor.DarkGray;
            Console.WriteLine($"  → temperature: {sampling.Temperature}");
            Console.ResetColor();
            continue;
        }
        if (line.StartsWith("/topk ") && int.TryParse(line[6..], out var newTopK))
        {
            sampling = sampling with { TopK = newTopK };
            Console.ForegroundColor = ConsoleColor.DarkGray;
            Console.WriteLine($"  → top-k: {sampling.TopK}");
            Console.ResetColor();
            continue;
        }
        if (line.StartsWith("/topp ") && float.TryParse(line[6..], out var newTopP))
        {
            sampling = sampling with { TopP = newTopP };
            Console.ForegroundColor = ConsoleColor.DarkGray;
            Console.WriteLine($"  → top-p: {sampling.TopP}");
            Console.ResetColor();
            continue;
        }

        prompt = line;
    }

    firstRun = false;
    var sw = System.Diagnostics.Stopwatch.StartNew();
    int tokenCount = 0;

    Console.ForegroundColor = ConsoleColor.Green;
    Console.Write($"» {prompt}");
    Console.ForegroundColor = ConsoleColor.White;

    await foreach (var token in GenerateStreamAsync(model, prompt, sampling, maxNewTokens: maxTokens))
    {
        Console.Write(token);
        tokenCount++;
    }

    sw.Stop();
    Console.WriteLine();
    Console.ForegroundColor = ConsoleColor.DarkGray;
    var tokPerSec = tokenCount / sw.Elapsed.TotalSeconds;
    Console.WriteLine($"  [{tokenCount} tokens in {sw.Elapsed.TotalSeconds:F1}s — {tokPerSec:F1} tok/s]");
    Console.ResetColor();
    Console.WriteLine();
}

return 0;

// =============================================================================
// Local functions
// =============================================================================

// ---- Argument parsing ----

static string? GetArg(string[] args, string name)
{
    for (int i = 0; i < args.Length - 1; i++)
        if (args[i] == name) return args[i + 1];
    return null;
}

static string? FindDefault(string filename, string relativeDir, bool returnDir = false)
{
    var dir = AppContext.BaseDirectory;
    for (int i = 0; i < 6; i++)
    {
        var candidate = Path.Combine(dir, relativeDir, filename);
        if (File.Exists(candidate))
            return returnDir ? Path.GetDirectoryName(candidate) : candidate;
        var parent = Directory.GetParent(dir)?.FullName;
        if (parent is null) break;
        dir = parent;
    }
    return null;
}

// ---- Tokenizer (BPE) ----
// NOTE: This hand-rolled BPE matches the custom tokenizer trained with gpt2-nano.
// For standard models (LLaMA, Phi, GPT), use Microsoft.ML.Tokenizers instead —
// it supports BPE, tiktoken, and SentencePiece out of the box.

static (List<string[]> merges, Dictionary<string, int> vocab) LoadTokenizer(string dir)
{
    var mergesJson = File.ReadAllText(Path.Combine(dir, "merges.json"));
    var rawMerges = JsonSerializer.Deserialize<List<List<string>>>(mergesJson)!;
    var mergesList = rawMerges.Select(m => new[] { m[0], m[1] }).ToList();

    var vocabJson = File.ReadAllText(Path.Combine(dir, "vocab.json"));
    var vocabDict = JsonSerializer.Deserialize<Dictionary<string, int>>(vocabJson)!;

    return (mergesList, vocabDict);
}

static List<int> Encode(string sentence, List<string[]> merges, Dictionary<string, int> vocab)
{
    const string eow = "<|EOW|>";
    // Split preserving empty entries to match Python's str.split(" ") behavior
    var words = sentence.Split(' ')
        .Where(w => w.Length > 0)  // Python's split(" ") drops only truly empty leading/trailing
        .Select(w => w.Select(c => c.ToString()).Append(eow).ToList())
        .ToList();

    foreach (var merge in merges)
    {
        var (a, b) = (merge[0], merge[1]);
        var merged = a + b;
        foreach (var word in words)
        {
            int i = 0;
            while (i < word.Count - 1)
            {
                if (word[i] == a && word[i + 1] == b)
                {
                    word[i] = merged;
                    word.RemoveAt(i + 1);
                }
                else i++;
            }
        }
    }

    return words.SelectMany(w => w.Select(t => vocab[t])).ToList();
}

// ---- Sampling ----

static Tensor SampleWithConfig(Tensor logits, SamplingConfig config)
{
    using var scope = torch.NewDisposeScope();

    var scaled = config.Temperature != 1.0f ? logits / config.Temperature : logits;
    var probs = torch.softmax(scaled, dim: -1);

    if (config.TopK > 0)
    {
        var (topkValues, _) = torch.topk(probs, config.TopK);
        var threshold = topkValues[-1];
        probs = torch.where(probs >= threshold, probs, torch.zeros_like(probs));
        probs = probs / probs.sum();
    }

    if (config.TopP < 1.0f)
    {
        var (sortedProbs, sortedIndices) = torch.sort(probs, descending: true);
        var cumProbs = torch.cumsum(sortedProbs, dim: -1);
        var mask = cumProbs - sortedProbs > config.TopP;
        sortedProbs = sortedProbs.masked_fill(mask, 0.0f);
        sortedProbs = sortedProbs / sortedProbs.sum();
        probs = torch.zeros_like(probs).scatter_(-1, sortedIndices, sortedProbs);
    }

    return torch.multinomial(probs, num_samples: 1).MoveToOuterDisposeScope();
}

// ---- Streaming generation ----

async IAsyncEnumerable<string> GenerateStreamAsync(
    GPT gpt, string prompt, SamplingConfig sampling,
    int maxNewTokens = 100,
    [EnumeratorCancellation] CancellationToken ct = default)
{
    var tokens = Encode(prompt, merges, vocab);
    using var noGrad = torch.no_grad();

    for (int i = 0; i < maxNewTokens; i++)
    {
        ct.ThrowIfCancellationRequested();
        using var scope = torch.NewDisposeScope();

        var window = tokens.Skip(Math.Max(0, tokens.Count - ContextLen)).ToArray();
        var x = torch.tensor(window, dtype: ScalarType.Int64).unsqueeze(0);
        var logits = gpt.forward(x);

        var nextTokenTensor = SampleWithConfig(logits[0, -1], sampling);
        var nextToken = (int)nextTokenTensor.item<long>();
        tokens.Add(nextToken);

        var text = inverseVocab.TryGetValue(nextToken, out var tok)
            ? (tok == "<|EOW|>" ? " " : tok) : "?";
        yield return text;

        await Task.Yield();
    }
}

// ---- SafeTensors loader with validation ----
// The SafeTensors format is data-only (JSON header + raw floats) — no code execution
// paths, no deserialization of arbitrary objects. Safe for loading untrusted weights.

const long MaxHeaderBytes = 100 * 1024 * 1024;
const long MaxTotalBytes = 10L * 1024 * 1024 * 1024;

static void LoadSafeTensors(GPT model, string path)
{
    var fileSize = new FileInfo(path).Length;
    if (fileSize > MaxTotalBytes)
        throw new InvalidOperationException($"Model file exceeds size limit ({fileSize} bytes).");

    var paramLookup = new Dictionary<string, Tensor>();
    foreach (var (pname, param) in model.named_parameters())
        paramLookup[pname] = param;

    using var noGrad = torch.no_grad();
    using var stream = File.OpenRead(path);
    using var reader = new BinaryReader(stream);

    // Parse and validate header
    if (fileSize < 8)
        throw new InvalidDataException("File too small to contain a SafeTensors header.");

    if (!path.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase))
        Console.Error.WriteLine($"  Warning: '{Path.GetFileName(path)}' doesn't have a .safetensors extension.");

    var headerLen = reader.ReadInt64();
    if (headerLen <= 0 || headerLen > MaxHeaderBytes || headerLen > fileSize - 8)
    {
        var msg = $"Not a valid SafeTensors file: '{Path.GetFileName(path)}'.";
        if (path.EndsWith(".pt", StringComparison.OrdinalIgnoreCase) ||
            path.EndsWith(".pth", StringComparison.OrdinalIgnoreCase) ||
            path.EndsWith(".bin", StringComparison.OrdinalIgnoreCase))
        {
            msg += "\n  This looks like a PyTorch checkpoint (.pt). Convert it to SafeTensors first:\n" +
                   "  python -c \"import torch; from safetensors.torch import save_file; " +
                   "cp = torch.load('" + Path.GetFileName(path) + "', map_location='cpu', weights_only=False); " +
                   "save_file({k: v.float() for k, v in cp['model_state_dict'].items()}, 'model.safetensors')\"";
        }
        throw new InvalidDataException(msg);
    }

    var headerBytes = reader.ReadBytes((int)headerLen);
    var header = JsonSerializer.Deserialize<Dictionary<string, JsonElement>>(headerBytes)
        ?? throw new InvalidDataException("Failed to parse SafeTensors header.");
    var dataOffset = 8 + headerLen;
    var dataRegionSize = fileSize - dataOffset;

    // Validate all entries before loading any data
    var entries = new List<(string name, string dtype, long[] shape, long byteStart, long byteEnd)>();
    foreach (var (name, meta) in header)
    {
        if (name == "__metadata__") continue;

        if (!meta.TryGetProperty("dtype", out var dtypeProp) ||
            !meta.TryGetProperty("shape", out var shapeProp) ||
            !meta.TryGetProperty("data_offsets", out var offsetsProp))
            throw new InvalidDataException($"Tensor '{name}': missing required fields.");

        var dtype = dtypeProp.GetString() ?? throw new InvalidDataException($"Tensor '{name}': null dtype.");
        var shape = shapeProp.EnumerateArray().Select(e => e.GetInt64()).ToArray();
        var offsets = offsetsProp.EnumerateArray().Select(e => e.GetInt64()).ToArray();
        if (offsets.Length != 2)
            throw new InvalidDataException($"Tensor '{name}': expected 2 offsets, got {offsets.Length}.");

        var (byteStart, byteEnd) = (offsets[0], offsets[1]);
        if (byteStart < 0 || byteEnd < byteStart || byteEnd > dataRegionSize)
            throw new InvalidDataException($"Tensor '{name}': offsets [{byteStart}, {byteEnd}] out of bounds.");

        if (dtype != "F32")
            throw new NotSupportedException(
                $"Tensor '{name}': dtype '{dtype}' is not supported. " +
                "Only F32 weights are accepted. Convert with: " +
                "save_file({{k: v.float() for k, v in sd.items()}}, 'model.safetensors')");

        const int bytesPerElement = 4;

        var numElements = shape.Length == 0 ? 1 : shape.Aggregate(1L, (a, b) => a * b);
        if (numElements * bytesPerElement != byteEnd - byteStart)
            throw new InvalidDataException($"Tensor '{name}': shape/dtype size mismatch.");

        entries.Add((name, dtype, shape, byteStart, byteEnd));
    }

    // Check for overlapping regions
    var sorted = entries.OrderBy(e => e.byteStart).ToList();
    for (int i = 1; i < sorted.Count; i++)
        if (sorted[i].byteStart < sorted[i - 1].byteEnd)
            throw new InvalidDataException($"Overlapping tensors: '{sorted[i - 1].name}' and '{sorted[i].name}'.");

    // Load validated tensors into model
    int loaded = 0;
    var loadedNames = new HashSet<string>();
    var posEmbeddingLoaded = false;
    foreach (var (name, dtype, shape, byteStart, byteEnd) in entries)
    {
        if (name.Contains("causal_mask")) continue;

        var numBytes = (int)(byteEnd - byteStart);
        stream.Position = dataOffset + byteStart;
        var data = reader.ReadBytes(numBytes);
        var floats = new float[numBytes / 4];
        Buffer.BlockCopy(data, 0, floats, 0, numBytes);
        var tensor = torch.tensor(floats, dimensions: shape);

        if (name == "pos_embedding")
        {
            model.SetPosEmbedding(tensor);
            posEmbeddingLoaded = true;
            loaded++;
            continue;
        }

        // Map Python attribute names → TorchSharp field names
        var mapped = name
            .Replace("input_embedding", "inputEmbedding")
            .Replace("attn_norm", "attnNorm")
            .Replace("mlp_norm", "mlpNorm")
            .Replace("lm_head", "lmHead");
        mapped = System.Text.RegularExpressions.Regex.Replace(mapped, @"mlp\.0\.", "mlpUp.");
        mapped = System.Text.RegularExpressions.Regex.Replace(mapped, @"mlp\.2\.", "mlpDown.");

        if (paramLookup.TryGetValue(mapped, out var param))
        {
            param.copy_(tensor);
            loadedNames.Add(mapped);
            loaded++;
        }
        else
        {
            Console.Error.WriteLine($"  Warning: unmatched weight '{name}' (mapped: '{mapped}')");
        }
    }
    // Verify all expected model parameters were loaded
    var missing = paramLookup.Keys.Where(k => !loadedNames.Contains(k)).ToList();
    if (!posEmbeddingLoaded)
        missing.Insert(0, "pos_embedding");
    if (missing.Count > 0)
        throw new InvalidDataException(
            $"Incomplete weights: {missing.Count} expected parameter(s) missing from '{Path.GetFileName(path)}':\n  " +
            string.Join("\n  ", missing));

    Console.WriteLine($"Loaded {loaded} tensors from {Path.GetFileName(path)}");
}

// =============================================================================
// Model architecture — matches the Python gpt2-nano implementation exactly.
// See https://github.com/kotlarmilos/gpt2-nano for the training code.
// =============================================================================

class Block : Module<Tensor, Tensor>
{
    private readonly int numHeads;
    private readonly int headDim;
    private readonly Linear Q, K, V;
    private readonly LayerNorm attnNorm, mlpNorm;
    private readonly Linear mlpUp, mlpDown;

    public Block(int embeddingDim, int contextLen, int numHeads)
        : base("Block")
    {
        this.numHeads = numHeads;
        headDim = embeddingDim / numHeads;

        Q = Linear(embeddingDim, embeddingDim);
        K = Linear(embeddingDim, embeddingDim);
        V = Linear(embeddingDim, embeddingDim);

        attnNorm = LayerNorm(embeddingDim);
        mlpNorm = LayerNorm(embeddingDim);
        mlpUp = Linear(embeddingDim, embeddingDim * 4);
        mlpDown = Linear(embeddingDim * 4, embeddingDim);

        RegisterComponents();
    }

    public override Tensor forward(Tensor inputs)
    {
        var (B, C, E) = (inputs.shape[0], inputs.shape[1], inputs.shape[2]);

        // Pre-norm + multi-head attention via SDPA (handles causal masking internally)
        var normed = attnNorm.forward(inputs);
        var q = Q.forward(normed).reshape(B, C, numHeads, headDim).transpose(1, 2);
        var k = K.forward(normed).reshape(B, C, numHeads, headDim).transpose(1, 2);
        var v = V.forward(normed).reshape(B, C, numHeads, headDim).transpose(1, 2);
        var attnOut = torch.nn.functional.scaled_dot_product_attention(q, k, v, null, 0.0, true);
        attnOut = attnOut.transpose(1, 2).reshape(B, C, E);
        var residual = inputs + attnOut;

        // Pre-norm MLP (512 → 2048 → 512)
        var mlpOut = mlpNorm.forward(residual);
        mlpOut = mlpUp.forward(mlpOut);
        mlpOut = torch.nn.functional.relu(mlpOut);
        mlpOut = mlpDown.forward(mlpOut);
        return residual + mlpOut;
    }
}

class GPT : Module<Tensor, Tensor>
{
    private readonly Embedding inputEmbedding;
    private readonly ModuleList<Block> blocks;
    private readonly Linear lmHead;
    private Tensor posEmbedding;

    public GPT(int vocabSize, int contextLen, int embeddingDim, int numLayers, int numHeads)
        : base("GPT")
    {
        inputEmbedding = Embedding(vocabSize, embeddingDim);
        posEmbedding = torch.zeros(1, contextLen, embeddingDim);

        blocks = new ModuleList<Block>();
        for (int i = 0; i < numLayers; i++)
            blocks.Add(new Block(embeddingDim, contextLen, numHeads));

        lmHead = Linear(embeddingDim, vocabSize);
        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        var seqLen = x.shape[1];
        var emb = inputEmbedding.forward(x) + posEmbedding[.., ..(int)seqLen, ..];
        foreach (var block in blocks)
            emb = block.forward(emb);
        return lmHead.forward(emb);
    }

    public void SetPosEmbedding(Tensor pe) => posEmbedding = pe;
}

record SamplingConfig(float Temperature = 1.0f, int TopK = 0, float TopP = 1.0f);
