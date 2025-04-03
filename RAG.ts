// import dotenv from 'dotenv';
// dotenv.config();
// import {
//     HarmCategory,
//     HarmBlockThreshold,
//     VertexAI,
//   } from "@google-cloud/vertexai";

// const vertex = new VertexAI({
//   project: process.env.VERTEXAI_PROJECT_ID,
//   location: process.env.VERTEXAI_LOCATION,
// });
// const model = process.env.VERTEXAI_MODEL_ID;
// const llm = vertex.getGenerativeModel({
//   model,
//   generationConfig: {
//     maxOutputTokens: 1000,
//     temperature: 1,
//     topP: 0.9,
//   }});

import { ChatVertexAI } from "@langchain/google-vertexai";

const llm = new ChatVertexAI({
  model: "gemini-2.0-flash",
  temperature: 0
});

import { VertexAIEmbeddings } from "@langchain/google-vertexai";

const embeddings = new VertexAIEmbeddings({
  model: "text-embedding-005"
});

import { MemoryVectorStore } from "langchain/vectorstores/memory";

const vectorStore = new MemoryVectorStore(embeddings);

import "cheerio";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { Document } from "@langchain/core/documents";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { pull } from "langchain/hub";

import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";


async function main() {
  // Load and chunk contents of blog
  const pTagSelector = "p";
  const cheerioLoader = new CheerioWebBaseLoader(
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    {
      selector: pTagSelector
    }
  );

  const docs = await cheerioLoader.load();

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000, chunkOverlap: 200
  });
  const allSplits = await splitter.splitDocuments(docs);

  // Index chunks
  await vectorStore.addDocuments(allSplits);

  // Define prompt for question-answering
  // const promptTemplate = await pull<ChatPromptTemplate>("rlm/rag-prompt");
  // let question = "What is Task Decomposition?";
  // const promptTemplate = ChatPromptTemplate.fromMessages([
  //   ["system", `You are a quiz generator. Your task is to generate exactly 5 quiz questions based on the context provided.
  //     Respond only in this format:
  //    [<question>]
  //    <<correct answer>>
  //    <<incorrect answer>>
  //    <<incorrect answer>>`],
  //   ["user", "{context}"],
  // ]);
  
  // const retrievedDocs = await vectorStore.similaritySearch(question);
  const docsContent = allSplits.map((doc) => doc.pageContent).join("\n");
  // const messages = await promptTemplate.invoke({
  //   context: docsContent,
  // });
  const contentRequest = {
    contents: [
      {
        role: "user",
        parts: [
          {
            text: `context: ${docsContent}`,
          },
        ],
      },
    ],
    systemInstruction: {
      role: "system",
      parts: [
        {
          text: `You are a quiz generator. Your task is to generate exactly 5 quiz questions based on the context provided.
      Respond only in this format:
     [<question>]
     {<correct answer>}
     {<incorrect answer>}
     {<incorrect answer>}`,
        },
      ],
    },
  };
  // console.log(`\nMessages: ${JSON.stringify(messages)}`);
  const result = await llm.generateContent(contentRequest);
  console.log(result.response.candidates[0].content);
}

main().catch(console.error);