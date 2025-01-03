import { FeatureExtractionPipeline, ImageFeatureExtractionPipeline, pipeline, PipelineType, RawImage, Tensor } from "@huggingface/transformers";

// Use the Singleton pattern to enable lazy construction of the pipeline.
class TextPipelineSingleton {
    static task: PipelineType = 'feature-extraction';
    static model = 'XudongShen/DFN-public';
    static instance: FeatureExtractionPipeline | null = null;

    static async getInstance(progress_callback = null) {
        this.instance ??= pipeline(this.task, this.model, {
            progress_callback,
            revision: "main",
            dtype: "fp32",
            model_file_name: "text_model",
          }) as unknown as FeatureExtractionPipeline;
        return this.instance;
    }
}
class ImagePipelineSingleton {
    static task:PipelineType = 'image-feature-extraction';
    static model = 'XudongShen/DFN-public';
    static instance: ImageFeatureExtractionPipeline | null = null;

    static async getInstance(progress_callback = null) {
        this.instance ??= pipeline(this.task, this.model, {
          progress_callback,
          revision: "main",
          dtype: "fp32",
          model_file_name: "vision_model",
        }) as unknown as ImageFeatureExtractionPipeline;
        return this.instance;
    }
}

// Listen for messages from the main thread
self.addEventListener('message', async (event) => {
    const textModel = await TextPipelineSingleton.getInstance(x => {
        self.postMessage(x);
    });

    const visionModel = await ImagePipelineSingleton.getInstance(x => {
        self.postMessage(x);
    });

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
        if (!textModel || !visionModel) {
          throw new Error("DFN not initialized. Call initialize() first.");
        }
    
        try {
          console.log("Computing similarity...", text, imageUrl);
    
          // Get text embeddings
          const textInputs: Tensor = textModel.tokenizer([text], {
            padding: "max_length", truncation: true
          });
          const textOutputs = await textModel.model(textInputs);
          const textEmbedding = normalize(textOutputs.text_embeds.ort_tensor.cpuData);
    
          // Get image embeddings

          const image = await RawImage.read(imageUrl);
          const imageInputs = await visionModel.processor([image]);
          const imageOutputs = await visionModel.model(imageInputs);
          const imageEmbedding = normalize( imageOutputs.image_embeds.ort_tensor.cpuData );
    
          // Compute cosine similarity
          return cosineSimilarity(textEmbedding, imageEmbedding);
        } catch (error) {
          console.error("Error computing similarity:", error);
          throw error;
        }
      }

    // Actually perform the classification
    const output = await computeSimilarity('a photo of a dog', 'https://cors-anywhere.herokuapp.com/https://place.dog/300/200');

    console.log("Output", output);
    // Send the output back to the main thread
    self.postMessage({
        status: 'complete',
        output: output,
    });
});
