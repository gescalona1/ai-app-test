import { pipeline, PipelineType } from "@xenova/transformers";

// Use the Singleton pattern to enable lazy construction of the pipeline.
// NOTE: We wrap the class in a function to prevent code duplication (see below).
const P = () => class PipelineSingleton {
    static task = 'feature-extraction';
    static model = 'Xenova/bge-small-en-v1.5';
    static instance: any = null;

    static async getInstance() {
        if (this.instance === null) {
            this.instance = pipeline(this.task as PipelineType, this.model);
        }
        return this.instance;
    }
}

let PipelineSingleton;
/*
if (process.env.NODE_ENV !== 'production') {
    // When running in development mode, attach the pipeline to the
    // global object so that it's preserved between hot reloads.
    // For more information, see https://vercel.com/guides/nextjs-prisma-postgres
    if (!(global as any).PipelineSingleton) {
        (global as any).PipelineSingleton = P();
    }
    PipelineSingleton = (global as any).PipelineSingleton;
} else*/ {
    PipelineSingleton = P();
}

// Export the PipelineSingleton type
export type PipelineSingletonType = typeof PipelineSingleton;

// Export the PipelineSingleton as the default export
export default PipelineSingleton as PipelineSingletonType;