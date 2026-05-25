import dotenv from 'dotenv';
dotenv.config();

async function checkModels() {
    console.log("🔍 Consultando a Google qué modelos tienes disponibles...");
    
    try {
        const response = await fetch(`https://generativelanguage.googleapis.com/v1beta/models?key=${process.env.GOOGLE_API_KEY}`);
        const data = await response.json();
        
        if (data.error) {
            console.error("❌ Error de API:", data.error.message);
            return;
        }

        // Filtramos SOLO los modelos que sirven para hacer embeddings (embedContent)
        const embedModels = data.models.filter(m => 
            m.supportedGenerationMethods && m.supportedGenerationMethods.includes('embedContent')
        );

        console.log("\n✅ MODELOS DE EMBEDDING DISPONIBLES PARA TU API KEY:");
        console.log("--------------------------------------------------");
        embedModels.forEach(m => {
            // Limpiamos el texto "models/" para darte el nombre exacto que necesitas
            console.log(`Nombre a usar: "${m.name.replace('models/', '')}"`);
            console.log(`Descripción: ${m.description}\n`);
        });
        console.log("--------------------------------------------------");
        
    } catch (err) {
        console.error("Error de conexión:", err);
    }
}

checkModels();