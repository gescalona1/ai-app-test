import { HuggingFaceEmbedding, OpenAIEmbedding, Settings, Document } from "llamaindex";

const MODEL_TYPE = "BAAI/bge-small-en-v1.5";
Settings.embedModel = new HuggingFaceEmbedding({
    modelType: MODEL_TYPE, // Replace with your chosen model
    quantized: false,
});

export const getEmbedding = async (text: string) => {
    return await Settings.embedModel.getTextEmbedding(text);
};