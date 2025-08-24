# micro_corn_diseases_detection
repositorio de microproyecto
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Informe: Detección de Enfermedades en Maíz</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary-color: #2c5282;
            --secondary-color: #2a69ac;
            --background-color: #f7fafc;
            --text-color: #4a5568;
            --card-background: #ffffff;
            --border-color: #e2e8f0;
            --shadow-color: rgba(0, 0, 0, 0.08);
            --font-family: 'Poppins', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }

        body {
            font-family: var(--font-family);
            background-color: var(--background-color);
            color: var(--text-color);
            margin: 0;
            padding: 2rem 1rem;
            line-height: 1.6;
        }

        .container {
            background: var(--card-background);
            padding: 2.5rem;
            border-radius: 16px;
            box-shadow: 0 10px 25px var(--shadow-color);
            max-width: 800px;
            margin: 0 auto;
        }

        h1, h2, h3 {
            color: var(--primary-color);
            font-weight: 600;
            border-bottom: 2px solid var(--border-color);
            padding-bottom: 0.5rem;
            margin-top: 2rem;
        }

        h1 {
            text-align: center;
            border-bottom: none;
            font-size: 2.5rem;
        }
        
        p, li {
            font-size: 1rem;
        }

        ul {
            padding-left: 20px;
        }
        
        strong {
            color: var(--secondary-color);
            font-weight: 600;
        }
        
        .section {
            margin-bottom: 2.5rem;
        }
        
        .subsection {
            margin-top: 1.5rem;
            padding-left: 1rem;
            border-left: 3px solid var(--secondary-color);
        }

        .highlight {
            background-color: #ebf8ff;
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid #bee3f8;
        }

    </style>
</head>
<body>
    <div class="container">
        <h1>🌽 Informe de Análisis Exploratorio</h1>
        <p style="text-align: center; font-size: 1.2rem; color: #718096;">Detección de Enfermedades en Cultivos de Maíz</p>
        
        <div class="section">
            <h2>1. Problema y Contexto</h2>
            <p>Las enfermedades del maíz, como la roya común, el tizón foliar y la mancha gris, representan una amenaza crítica para la seguridad alimentaria. El diagnóstico tradicional mediante inspección visual es un proceso lento, subjetivo y dependiente de la pericia del observador. Este proyecto busca validar la viabilidad de un sistema de diagnóstico automatizado mediante Inteligencia Artificial para superar estas limitaciones.</p>
        </div>
        
        <div class="section">
            <h2>2. Descripción del Dataset Inicial</h2>
            <p>Para el análisis, se utilizó el dataset público "Corn or Maize Leaf Disease Dataset" de Kaggle, una compilación de imágenes de las fuentes PlantVillage y PlantDoc.</p>
            <ul>
                <li><strong>Total de Imágenes:</strong> 4,188</li>
                <li><strong>Formato:</strong> JPEG (.jpg)</li>
                <li><strong>Distribución de Clases (Inicial):</strong>
                    <ul>
                        <li>Roya Común (Common Rust): 1,306 imágenes (31.2%)</li>
                        <li>Sana (Healthy): 1,162 imágenes (27.7%)</li>
                        <li>Tizón (Blight): 1,146 imágenes (27.4%)</li>
                        <li>Mancha Gris (Gray Leaf Spot): 574 imágenes (13.7%)</li>
                    </ul>
                </li>
            </ul>
            <p class="highlight"><strong>Observación Clave:</strong> El dataset inicial presenta un notable desbalance, con la clase "Mancha Gris" significativamente subrepresentada. Este hallazgo es fundamental para las siguientes etapas del proyecto.</p>
        </div>

        <div class="section">
            <h2>3. Hallazgos del Análisis Exploratorio de Datos (EDA)</h2>
            
            <div class="subsection">
                <h3>3.1. Validación e Integridad de Datos</h3>
                <p>Se realizó una validación estructural del dataset para confirmar la cantidad de clases, el número de imágenes y la integridad de los archivos. Se encontró y corrigió una inconsistencia de formato (un archivo <strong>.jpeg</strong> en lugar de <strong>.jpg</strong>) en la clase "Blight", asegurando la homogeneidad del conjunto de datos.</p>
            </div>
            
            <div class="subsection">
                <h3>3.2. Análisis Cualitativo Visual</h3>
                <p>La inspección de muestras aleatorias reveló una buena calidad de imagen general (nitidez y enfoque). Se destacó una alta variabilidad en iluminación, escala y ángulos de captura, lo cual es beneficioso para entrenar un modelo más robusto y generalizable.</p>
                <p class="highlight"><strong>Desafío Principal Identificado:</strong> Se observó una alta similitud morfológica entre las lesiones en etapas avanzadas de <strong>"Mancha Gris"</strong> y <strong>"Tizón"</strong>, lo que anticipa el principal reto de clasificación para el modelo de IA.</p>
            </div>

            <div class="subsection">
                <h3>3.3. Análisis Cuantitativo de Características Físicas</h3>
                <ul>
                    <li><strong>Dimensiones:</strong> Se confirmó una considerable variabilidad en el tamaño (alto y ancho) de las imágenes, lo que fundamenta la necesidad de un paso de redimensionamiento estándar antes de alimentar el modelo.</li>
                    <li><strong>Distribución de Color:</strong> El análisis de histogramas de color, particularmente en el canal verde, demostró ser un rasgo altamente discriminatorio. Las hojas sanas ("Healthy") mostraron un perfil de color verde único y vibrante, claramente distinto al de las hojas enfermas. Esto valida el potencial del color como una característica potente para la clasificación automática y justifica la necesidad de normalizar los valores de los píxeles.</li>
                </ul>
            </div>
        </div>
    </div>
</body>
</html>