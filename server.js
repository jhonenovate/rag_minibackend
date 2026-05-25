import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { GoogleGenerativeAI } from "@google/generative-ai";

dotenv.config();

const app = express();
app.use(cors());
app.use(express.json());

// Configuraciones de ruta
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Inicializar Google SDK (El mismo que usabas en Frontend)
const genAI = new GoogleGenerativeAI(process.env.GOOGLE_API_KEY);
// Usaremos el modelo de embeddings más reciente

const embeddingModel = genAI.getGenerativeModel({ model: "gemini-embedding-001" });
// Usaremos Flash para responder
const chatModel = genAI.getGenerativeModel({ model: "gemini-2.5-flash" });

// ---------------------------------------------------------
// EL CORAZÓN DEL RAG: NUESTRA BD VECTORIAL Y MATEMÁTICAS
// ---------------------------------------------------------
let vectorStore = []; // Aquí guardaremos [{text: "...", vector: [0.1, 0.5...]}]
const RAG_DOCS_PATH = 'C:/Users/jsda0/Desktop/Enovate/Production Analytics/frontend_dashboardprodanalytics/rag_docs';
const delay = (ms) => new Promise(resolve => setTimeout(resolve, ms));
// Función matemática: Similitud del Coseno
console.log(RAG_DOCS_PATH);

// Compara dos arrays de 768 dimensiones y devuelve un % de similitud
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

// ---------------------------------------------------------
// FASE 1: INGESTA (Leer, Partir y Vectorizar)
// ---------------------------------------------------------
async function initializeRAG() {
 console.log("🧠 Inicializando el Cerebro RAG Nativo...");
 const chunks = [];

 // 1. Leer archivos .md de tu frontend (Filtrando carpetas innecesarias)
 const readFilesRecursively = (dir) => {
  // 🛑 EL FILTRO: Si la ruta de la carpeta incluye "guia_es", nos salimos y no la leemos
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
    // También omitimos cualquier archivo README.md por si acaso
    if (file.toLowerCase() === 'readme.md') continue;

    const text = fs.readFileSync(fullPath, 'utf8');

    // 💡 NUEVO: Extraer el nombre del módulo usando una Expresión Regular simple
    let docModuleName = "Unknown Module";
    const moduleMatch = text.match(/module:\s*"([^"]+)"/);
    if (moduleMatch && moduleMatch[1]) {
     docModuleName = moduleMatch[1];
    }

    // 2. Fragmentación Inteligente (Chunking)
    const splitTexts = text.split('\n## ');
    for (let t of splitTexts) {
     if (t.trim().length > 50) {
      // 💡 NUEVO: Le inyectamos el nombre del módulo a CADA fragmento
      const chunkWithMetadata = `[ORIGIN MODULE: ${docModuleName}]\n\n${t.trim()}`;
      chunks.push(chunkWithMetadata);
     }
    }
   }
  }
 };

 readFilesRecursively(RAG_DOCS_PATH);
 console.log(`🔪 Documentos divididos en ${chunks.length} fragmentos lógicos.`);

 console.log("\n👀 VISTAZO RÁPIDO A LOS PRIMEROS 2 FRAGMENTOS QUE SE VAN A VECTORIZAR:");
 console.log("--------------------------------------------------");
 console.log(chunks[0]);
 console.log("--------------------------------------------------");
 if (chunks.length > 1) console.log(chunks[1]);
 console.log("--------------------------------------------------\n");

 // 3. Obtener Embeddings directamente de Google
 // 3. Obtener Embeddings (Ajustado a la Capa Gratuita de Google: 15 RPM)
 console.log("🔢 Vectorizando fragmentos a velocidad de capa gratuita (aprox 4.5 seg por fragmento)...");

 for (let i = 0; i < chunks.length; i++) {
  let success = false;
  let attempts = 0;

  // Intentará hasta 3 veces por cada fragmento antes de rendirse
  while (!success && attempts < 3) {
   try {
    const result = await embeddingModel.embedContent(chunks[i]);
    const vector = result.embedding.values;

    vectorStore.push({ text: chunks[i], vector: vector });
    console.log(`✅ Fragmento ${i + 1}/${chunks.length} vectorizado.`);

    success = true;
    await delay(500);

   } catch (err) {
    attempts++;
    console.error(`\n🚨 ERROR REAL DE GOOGLE (Fragmento ${i + 1}):`, err.message);
    console.warn(`⚠️ Intento ${attempts}/3. Pausando 5 segundos...`);
    await delay(5000);
   }
  }

  if (!success) {
   console.error(`❌ Se omitió el fragmento ${i + 1} tras 3 intentos fallidos.`);
  }
 }
 console.log("✅ Base de Datos Vectorial Nativa ¡LISTA!");
}

// ---------------------------------------------------------
// FASE 2: RECUPERACIÓN Y GENERACIÓN (El Endpoint)
// ---------------------------------------------------------
app.post('/api/chat', async (req, res) => {
 try {
  const { query } = req.body;
  if (vectorStore.length === 0) {
   return res.status(500).json({ error: "El sistema RAG aún está cargando." });
  }

  console.log(`\n🗣️ Usuario pregunta: "${query}"`);

  const { userQuestion, systemContext } = req.body;
  // 1. Convertir la pregunta del usuario en un Vector (Números)
  const queryResult = await embeddingModel.embedContent(userQuestion);
  const queryVector = queryResult.embedding.values;

  // ⚠️ SUGERENCIA: Comentado para evitar que colapse la terminal por exceso de datos
  // console.log(vectorStore, 'vectorStore');

  // 2. Retrieval (Búsqueda por Similitud)
  const scoredDocs = vectorStore.map(doc => {
   return {
    text: doc.text,
    score: cosineSimilarity(queryVector, doc.vector)
   };
  });

  // Ordenamos de mayor a menor similitud y tomamos los top 9
  scoredDocs.sort((a, b) => b.score - a.score);
  const topDocs = scoredDocs.slice(0, 9);

  const contextText = topDocs.map(d => d.text).join("\n\n---\n\n");
  console.log(`🔍 Se inyectaron los ${topDocs.length} fragmentos con mayor similitud.`);

  // 🔧 AJUSTE 2: Nuevo Prompt Inteligente
  const prompt = `
            Eres el "Technical AI Copilot" experto en la plataforma. Hablas de forma profesional, clara y orientada a operadores o ingenieros.
            
            REGLAS DE ORO:
            1. El mensaje del usuario contiene un bloque "[System Context...]" y su "User Question".
            2. Usa el "System Context" para responder en qué módulo/pozo estamos o valores actuales (KPIs).
            3. Usa la "DOCUMENTACIÓN OFICIAL" para responder.
            4. 🚫 REGLA DE ABSTRACCIÓN TÉCNICA: No menciones variables, componentes ni constantes. Traduce a lenguaje de negocio.
            5. 🛡️ REGLA DE CONGRUENCIA DE MÓDULO (ESTRICTA): Cada fragmento de la documentación empieza con la etiqueta "[ORIGIN MODULE: X]". Compara esa 'X' con el "Module" de la situación actual del usuario. Si NO son exactamente el mismo módulo, ES OBLIGATORIO comenzar tu respuesta diciendo textualmente: "Actualmente te encuentras en el módulo [Módulo del Usuario] y no tengo documentación específica para este. Sin embargo, en el módulo [Módulo de Origen] funciona de la siguiente manera, aunque podría variar:". NUNCA asumas que aplica para el módulo actual.
            6. Si la respuesta definitiva no está en los documentos, di amablemente que no tienes esa información.

            DOCUMENTACIÓN OFICIAL RECUPERADA:
            ${contextText}

            [SITUACIÓN ACTUAL DEL USUARIO EN LA PLATAFORMA]
            ${systemContext}
            
            PREGUNTA DEL USUARIO:
            ${userQuestion}
            
            RESPUESTA:
        `;

  // 4. Generación (Llamar a Gemini con el contexto EN MODO STREAMING)
  // Configuramos las cabeceras HTTP para Server-Sent Events (SSE)
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');

  // Usamos generateContentStream en lugar de generateContent
  const result = await chatModel.generateContentStream(prompt);

  // Iteramos sobre los pedacitos (chunks) que nos va devolviendo Google en vivo
  for await (const chunk of result.stream) {
   const chunkText = chunk.text();
   // Enviamos el pedacito al frontend usando el formato estándar de SSE
   res.write(`data: ${JSON.stringify({ text: chunkText })}\n\n`);
  }

  // Le avisamos al frontend que terminamos de enviar el stream
  res.write(`data: [DONE]\n\n`);
  res.end();

 } catch (error) {
  console.error("Error en RAG:", error);
  // Manejo de error seguro por si el stream ya había comenzado
  if (!res.headersSent) {
   res.status(500).json({ error: "Error procesando la solicitud." });
  } else {
   res.write(`data: ${JSON.stringify({ text: "\n\n[Error de conexión con el modelo en medio de la respuesta]" })}\n\n`);
   res.write(`data: [DONE]\n\n`);
   res.end();
  }
 }
});

// ---------------------------------------------------------
// INICIO DEL SERVIDOR
// ---------------------------------------------------------
const PORT = process.env.PORT || 3001;

app.listen(PORT, async () => {
 console.log(`🚀 Servidor Express corriendo en http://localhost:${PORT}`);
 // Inicializamos el RAG antes de aceptar peticiones
 await initializeRAG();
});