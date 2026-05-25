import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { GoogleGenerativeAI } from "@google/generative-ai";
// 1. Importamos la librería Open Source para IA local
import { pipeline } from '@xenova/transformers';

dotenv.config();

const app = express();
app.use(cors());
app.use(express.json());

// Configuraciones de ruta
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// 2. Mantenemos Google SDK SOLO para el Chat
const genAI = new GoogleGenerativeAI(process.env.GOOGLE_API_KEY);
const chatModel = genAI.getGenerativeModel({ model: "gemini-2.5-flash" });

// ----------------─────────────────────────────────────────
// EL CORAZÓN DEL RAG: NUESTRA BD VECTORIAL Y MATEMÁTICAS
// ----------------─────────────────────────────────────────
let vectorStore = [];
let extractor; // Aquí vivirá nuestro modelo local de Hugging Face
const RAG_DOCS_PATH = 'C:/Users/jsda0/Desktop/Enovate/Production Analytics/frontend_dashboardprodanalytics/rag_docs';

console.log(RAG_DOCS_PATH);

// Función matemática: Similitud del Coseno (Funciona igual sin importar la dimensión)
function cosineSimilarity(vecA, vecB) {
 let dotProduct = 0;
 let normA = 0;
 let normB = 0;
 for (let i = 0; i < vecA.length; i++) {
  dotProduct += vecA[i] * vecB[i];
  normA += vecA[i] * vecA[i];
  normB += vecB[i] * vecB[i];
 }
 if (normA === 0 || normB === 0) return 0;
 return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

// ----------------─────────────────────────────────────────
// FASE 1: INGESTA (Leer, Partir y Vectorizar OFFLINE)
// ----------------─────────────────────────────────────────
async function initializeRAG() {
 console.log("🧠 Inicializando el Cerebro RAG Nativo...");

 // 3. Inicializamos el modelo de Embeddings Local (Hugging Face)
 console.log("📥 Cargando modelo de Embeddings local (all-MiniLM-L6-v2)...");
 extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
 console.log("✅ Modelo local cargado en memoria.");

 const chunks = [];

 const readFilesRecursively = (dir) => {
  if (dir.includes('guia_es')) {
   console.log(`⏭️ Omitiendo carpeta exclusiva para Devs: ${dir}`);
   return;
  }

  const files = fs.readdirSync(dir);
  for (const file of files) {
   const fullPath = path.join(dir, file);

   if (fs.statSync(fullPath).isDirectory()) {
    readFilesRecursively(fullPath);
   } else if (fullPath.endsWith('.md')) {
    if (file.toLowerCase() === 'readme.md') continue;

    const text = fs.readFileSync(fullPath, 'utf8');

    let docModuleName = "Unknown Module";
    const moduleMatch = text.match(/module:\s*"([^"]+)"/);
    if (moduleMatch && moduleMatch[1]) {
     docModuleName = moduleMatch[1];
    }

    const splitTexts = text.split('\n## ');
    for (let t of splitTexts) {
     if (t.trim().length > 50) {
      const chunkWithMetadata = `[ORIGIN MODULE: ${docModuleName}]\n\n${t.trim()}`;
      chunks.push(chunkWithMetadata);
     }
    }
   }
  }
 };

 readFilesRecursively(RAG_DOCS_PATH);
 console.log(`🔪 Documentos divididos en ${chunks.length} fragmentos lógicos.`);

 // 4. Vectorización superrápida (Sin delays, sin límites de API)
 console.log("🔢 Vectorizando fragmentos 100% OFFLINE y GRATIS...");
 for (let i = 0; i < chunks.length; i++) {
  try {
   // Generamos el vector usando Transformers.js
   const output = await extractor(chunks[i], { pooling: 'mean', normalize: true });
   // Convertimos el Float32Array a un Array nativo de JS
   const vector = Array.from(output.data);

   vectorStore.push({ text: chunks[i], vector: vector });
   console.log(`✅ Fragmento ${i + 1}/${chunks.length} vectorizado en local.`);
  } catch (err) {
   console.error(`❌ Error vectorizando el fragmento ${i + 1}:`, err.message);
  }
 }
 console.log("✅ Base de Datos Vectorial Nativa ¡LISTA!");
}

// ----------------─────────────────────────────────────────
// FASE 2: RECUPERACIÓN Y GENERACIÓN (El Endpoint)
// ----------------─────────────────────────────────────────
app.post('/api/chat', async (req, res) => {
 try {
  const { userQuestion, systemContext } = req.body;
  if (vectorStore.length === 0) {
   return res.status(500).json({ error: "El sistema RAG aún está cargando." });
  }

  console.log(`\n🗣️ Usuario pregunta: "${userQuestion}"`);

  // ----------------─────────────────────────────────────────
  // 5. ENRIQUECIMIENTO DE QUERY (Query Enrichment)
  // ----------------─────────────────────────────────────────
  // Extraemos el nombre de la vista actual desde el systemContext
  const viewMatch = systemContext.match(/View:\s*([^,]+)/);
  const currentView = viewMatch ? viewMatch[1].trim() : "";

  // Inyectamos la vista a la pregunta ANTES de pasarla al modelo de embeddings local
  const enrichedSearchQuery = currentView
   ? `Regarding the module or view "${currentView}": ${userQuestion}`
   : userQuestion;

  console.log(`🔍 Query enriquecida para el motor vectorial: "${enrichedSearchQuery}"`);

  // Convertimos la pregunta enriquecida a Vector localmente
  const output = await extractor(enrichedSearchQuery, { pooling: 'mean', normalize: true });
  const queryVector = Array.from(output.data);

  // ----------------─────────────────────────────────────────
  // 6. Retrieval (Búsqueda por Similitud)
  // ----------------─────────────────────────────────────────
  const scoredDocs = vectorStore.map(doc => {
   return {
    text: doc.text,
    score: cosineSimilarity(queryVector, doc.vector)
   };
  });

  scoredDocs.sort((a, b) => b.score - a.score);
  const topDocs = scoredDocs.slice(0, 9);

  const contextText = topDocs.map(d => d.text).join("\n\n---\n\n");
  console.log(`🔍 Se inyectaron los ${topDocs.length} fragmentos con mayor similitud.`);

  // Prompt (Usamos la pregunta original 'userQuestion' para que Gemini no se confunda)
  const prompt = `
            Eres el "Technical AI Copilot" experto en la plataforma. Hablas de forma profesional, clara y orientada a operadores o ingenieros de la industria.
            
            REGLAS DE ORO:
            1. El mensaje del usuario contiene un bloque "[System Context...]" y su "User Question".
            2. Usa el "System Context" para responder. NUNCA uses el nombre del "Module" para saludar o referirte a la ubicación del usuario; usa SIEMPRE el nombre de la "View" (ej. "En la vista Choke Performance...").
            3. 🚫 REGLA DE INMERSIÓN: NUNCA uses frases como "según la documentación", "el manual indica", "la documentación oficial describe" o "los documentos dicen". Actúa con autoridad y propiedad; el conocimiento es tuyo.
            4. ⚡ REGLA DE CONCISIÓN: Ve directo al punto. Evita introducciones largas, saludos repetitivos, o conclusiones innecesarias. Si hay varias reglas o pasos, usa viñetas (bullet points) para que sea fácil de leer.
            5. 🚫 REGLA DE ABSTRACCIÓN TÉCNICA: No menciones variables (ej. isCondensate), componentes de código ni constantes. Traduce todo a lenguaje de negocio.
            6. 🛡️ REGLA DE CONGRUENCIA DE MÓDULO (ESTRICTA): Cada fragmento de la documentación empieza con la etiqueta "[ORIGIN MODULE: X]". Compara esa 'X' con el "Module" de la situación actual del usuario. Si NO son exactamente el mismo módulo, ES OBLIGATORIO comenzar tu respuesta diciendo textualmente: "Actualmente te encuentras en la vista [View del Usuario] y no tengo documentación específica para el módulo al que pertenece. Sin embargo, en el módulo [Módulo de Origen] funciona de la siguiente manera, aunque podría variar:". NUNCA asumas que aplica para el módulo actual si no coinciden.
            7. Si la respuesta no está en el contexto recuperado, indícalo amablemente de forma directa.

            CONTEXTO RECUPERADO:
            ${contextText}

            [SITUACIÓN ACTUAL DEL USUARIO EN LA PLATAFORMA]
            ${systemContext}
            
            PREGUNTA DEL USUARIO:
            ${userQuestion}
            
            RESPUESTA:
        `;

  // 7. Generación (La única parte que sí consume Google API)
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');

  const result = await chatModel.generateContentStream(prompt);

  for await (const chunk of result.stream) {
   const chunkText = chunk.text();
   res.write(`data: ${JSON.stringify({ text: chunkText })}\n\n`);
  }

  res.write(`data: [DONE]\n\n`);
  res.end();

 } catch (error) {
  console.error("Error en RAG:", error);
  if (!res.headersSent) {
   res.status(500).json({ error: "Error procesando la solicitud." });
  } else {
   res.write(`data: ${JSON.stringify({ text: "\n\n[Error de conexión con el modelo en medio de la respuesta]" })}\n\n`);
   res.write(`data: [DONE]\n\n`);
   res.end();
  }
 }
});

// ----------------─────────────────────────────────────────
// INICIO DEL SERVIDOR
// ----------------─────────────────────────────────────────
const PORT = process.env.PORT || 3001;

app.listen(PORT, async () => {
 console.log(`🚀 Servidor Express corriendo en http://localhost:${PORT}`);
 await initializeRAG();
});