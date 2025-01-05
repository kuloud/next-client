import { AutoProcessor, AutoTokenizer, CLIPTextModelWithProjection, CLIPVisionModelWithProjection, FeatureExtractionPipeline, ImageFeatureExtractionPipeline, pipeline, PipelineType, PreTrainedModel, PreTrainedTokenizer, Processor, RawImage, Tensor } from "@huggingface/transformers";

class DFN {
  private processor?: Processor;
  private tokenizer?: PreTrainedTokenizer;
  private textModel?: PreTrainedModel;
  private visionModel?: PreTrainedModel;

  constructor() {}

  async initialize(modelName: string = 'XudongShen/DFN-public', progress_callback = null) {
    this.processor = await AutoProcessor.from_pretrained(modelName, {});
    this.tokenizer = await AutoTokenizer.from_pretrained(modelName);
    this.textModel = await CLIPTextModelWithProjection.from_pretrained(modelName, {
      progress_callback,
      dtype: "fp16",
    });
    this.visionModel = await CLIPVisionModelWithProjection.from_pretrained(modelName, {
      progress_callback,
      dtype: "fp16",
    });
    return this;
  }

  getProcessor() {
    return this.processor;
  }

  getTokenizer() {
    return this.tokenizer;
  }

  getTextModel() {
    return this.textModel;
  }

  getVisionModel() {
    return this.visionModel;
  }
}

// Use the Singleton pattern to enable lazy construction of the pipeline.
class PipelineSingleton {
    static model = 'XudongShen/DFN-public';
    static instance: DFN | null = null;

    static async getInstance(progress_callback = null) {
        this.instance ??= await new DFN().initialize(this.model, progress_callback);
        return this.instance;
    }
}

// Listen for messages from the main thread
self.addEventListener('message', async (event) => {
    const dfn = await PipelineSingleton.getInstance(x => self.postMessage({ status: 'progress', progress: x }));

    console.log("DFN initialized");

    const cosineSimilarity = (vecA, vecB) => {
        let dot = 0.0;
        let normA = 0.0;
        let normB = 0.0;
    
        for (let i = 0; i < vecA.length; i++) {
          dot += vecA[i] * vecB[i];
          normA += vecA[i] * vecA[i];
          normB += vecB[i] * vecB[i];
        }
    
        return dot / (Math.sqrt(normA) * Math.sqrt(normB));
      }

      function normalize(vec: Float32Array): Float32Array {
        let norm = 0.0;
        for (let i = 0; i < vec.length; i++) {
          norm += vec[i] * vec[i];
        }
        norm = Math.sqrt(norm);
      
        const result = new Float32Array(vec.length);
        for (let i = 0; i < vec.length; i++) {
          result[i] = vec[i] / norm;
        }
      
        return result;
      }

    const computeSimilarity = async (text: string, imageUrl: string) => {
        if (!dfn) {
          throw new Error("DFN not initialized. Call initialize() first.");
        }
    
        try {
          console.log("Computing similarity...", text, imageUrl);
    
          // Get text embeddings
          // Get tokenizer max length
          const maxLength = dfn.getTokenizer().model_max_length; // default CLIP length is 77

          // Get complete token sequence without truncation
          const tokens = dfn.getTokenizer()([text], {
              truncation: false,
              padding: false
          }).input_ids.ort_tensor.cpuData;

          // Split into chunks
          const tokenChunks = [];
          for (let i = 0; i < tokens.length; i += maxLength) {
              tokenChunks.push(tokens.slice(i, i + maxLength));
          }

          // Process each chunk and calculate embeddings
          const chunkEmbeddings = [];
          for (const chunk of tokenChunks) {
            if (chunk.length === 0) {
                continue;
            }
            const chunkBigIntArray = Array.from(chunk, BigInt)
              // Convert tokens back to text
              const chunkText = dfn.getTokenizer().decode(chunkBigIntArray, {
                  skip_special_tokens: true
              });

              // Get chunk embedding
              const textInputs = dfn.getTokenizer()([chunkText], {
                  padding: "max_length",
                  truncation: true
              });
              const textOutputs = await dfn.getTextModel()(textInputs);
              const chunkEmbedding = normalize(textOutputs.text_embeds.ort_tensor.cpuData);
              chunkEmbeddings.push(chunkEmbedding);
          }

          // Calculate mean embedding
          let textEmbedding: Float32Array;
          if (chunkEmbeddings.length === 1) {
              textEmbedding = chunkEmbeddings[0];
          } else {
              // Calculate mean of all chunk embeddings
              textEmbedding = new Float32Array(chunkEmbeddings[0].length);
              for (const embedding of chunkEmbeddings) {
                  for (let i = 0; i < embedding.length; i++) {
                      textEmbedding[i] += embedding[i] / chunkEmbeddings.length;
                  }
              }
              // Renormalize mean embedding if multiple chunks
              textEmbedding = normalize(textEmbedding);
          }
          console.log('textEmbedding', textEmbedding)
    
          // Get image embeddings

          const image = await RawImage.read(imageUrl);
          const imageInputs = await dfn.getProcessor()([image]);
          const imageOutputs = await dfn.getVisionModel()(imageInputs);
          const imageEmbedding = normalize( imageOutputs.image_embeds.ort_tensor.cpuData );
    
          // Compute cosine similarity
          return cosineSimilarity(textEmbedding, imageEmbedding);
        } catch (error) {
          console.error("Error computing similarity:", error);
          throw error;
        }
      }

    // Actually perform the classification
    const output = await computeSimilarity('Customizing Windows 7 Setup Please Help Solved', 'https://i.imgur.com/mXQrfNs.png');

    console.log("Output", output);
    // Send the output back to the main thread
    self.postMessage({
        status: 'complete',
        output: output,
    });
});
