// Document loader
import { CheerioWebBaseLoader } from "langchain/document_loaders/web/cheerio";
// split document
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
//embed and store
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";

// chat
import { RetrievalQAChain } from "langchain/chains";
import { ChatOpenAI } from "langchain/chat_models/openai";

// pdf
import { PDFLoader } from "langchain/document_loaders/fs/pdf";

// const loader = new CheerioWebBaseLoader(
//   "https://lilianweng.github.io/posts/2023-06-23-agent/"
// );
// const data = await loader.load();

const loader = new PDFLoader("./example_doc.pdf");

const data = await loader.load();

// split document
const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 500,
    chunkOverlap: 0,
  });
  
  const splitDocs = await textSplitter.splitDocuments(data);
  

// Embed and store the splits in a vector database 
// (for demo purposes we use an unoptimized, in-memory example but you can browse integrations here):

const embeddings = new OpenAIEmbeddings({verbose:true, openAIApiKey: "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" });

const vectorStore = await MemoryVectorStore.fromDocuments(splitDocs, embeddings);

//

// const relevantDocs = await vectorStore.similaritySearch("What is task decomposition?");

// console.log(relevantDocs.length);

// Getting Started
const model = new ChatOpenAI({ modelName: "gpt-3.5-turbo", openAIApiKey: "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" });
const chain = RetrievalQAChain.fromLLM(model, vectorStore.asRetriever());

const response = await chain.call({
  query: "Que palabras estan entre llaves?"
});

console.log(response);