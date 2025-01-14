import { NextResponse } from 'next/server'
import { Pinecone } from '@pinecone-database/pinecone'
import OpenAI from 'openai'
import PipelineSingleton from '@/app/api/chat/pipeline.tsx';

const systemPrompt = `
You are a rate my professor agent to help students find classes, that takes in user questions and answers them.
For every user question, the top 3 professors that match the user question are returned.
Use them to answer the question if needed.`

export const POST = async (req: Request) => {
    const data = await req.json();
    const pc = new Pinecone({
        apiKey: process.env.PINECONE_API_KEY ?? '',
    });
    const index = pc.index('rag').namespace('ns1');
    const openai = new OpenAI({
        baseURL: "https://openrouter.ai/api/v1",
        apiKey: process.env.OPENROUTER_API_KEY ?? '',
      })

    const text = [data[data.length - 1].content];
    const extractor = await PipelineSingleton.getInstance();
    const result = await extractor(text, { pooling: 'mean', normalize: true });
    const embedding = result.tolist();
    

    const results = await index.query({
        topK: 5,
        includeMetadata: true,
        vector: embedding,
    });
    
    const resultString = results.matches.reduce((acc, match) => 
        acc + `
        Returned Results:
        Professor: ${match?.id}
        Review: ${match.metadata?.review}
        Subject: ${match.metadata?.subject}
        Stars: ${match.metadata?.stars}
        \n\n`, '');
    const lastMessage = data[data.length - 1];
    const lastMessageContent = lastMessage.content + resultString;
    const lastDataWithoutLastMessage = data.slice(0, data.length - 1);

    const completion = await openai.chat.completions.create({
        messages: [
            {role: 'system', content: systemPrompt},
            ...lastDataWithoutLastMessage,
            {role: 'user', content: lastMessageContent},
        ],
        model: 'meta-llama/llama-3.1-8b-instruct:free',
        stream: true,
    });

    const stream = new ReadableStream({
        async start(controller) {
            const encoder = new TextEncoder()
            try {
                for await (const chunk of completion) {
                    const content = chunk.choices[0]?.delta?.content
                    if (content) {
                        const text = encoder.encode(content)
                        controller.enqueue(text)
                    }
                }
            } catch (err) {
                controller.error(err)
            } finally {
                controller.close()
            }
        },
    })
    return new NextResponse(stream)
}