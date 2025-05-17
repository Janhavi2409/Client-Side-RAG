import React, { useState, useEffect, useRef } from 'react';
import './App.css';
import * as Papaparse from 'papaparse';
import _ from 'lodash';

// Main App component
function App() {
  const [documents, setDocuments] = useState([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');
  const [loading, setLoading] = useState(false);
  const [modelLoading, setModelLoading] = useState(false);
  const [modelStatus, setModelStatus] = useState('Not loaded');
  const [selectedModel, setSelectedModel] = useState('llama2');
  const [embeddingsModel, setEmbeddingsModel] = useState(null);
  const [searchType, setSearchType] = useState('semantic');
  const fileInputRef = useRef(null);
  const [error, setError] = useState('');
  const [availableModels, setAvailableModels] = useState([]);
  const [memoryLimit, setMemoryLimit] = useState(null);
  const [useChunking, setUseChunking] = useState(true);
  const [chunkSize, setChunkSize] = useState(500);
  const [chunkOverlap, setChunkOverlap] = useState(50);

  // Initialize the component when it mounts
  useEffect(() => {
    checkOllamaStatus();
  }, []);

  // Check if Ollama is running and available
  const checkOllamaStatus = async () => {
    try {
      const response = await fetch('http://localhost:11434/api/tags');
      if (response.ok) {
        const data = await response.json();
        setModelStatus('Ollama server is running');

        // Store available models
        if (data.models && data.models.length > 0) {
          setAvailableModels(data.models.map(model => model.name));

          // List of embedding models from smallest to largest
          const embeddingModels = [
            'all-minilm:1.5b', // Smallest, ~1.5GB
            'e5-small-v2:1.3b',
            'all-minilm',
            'nomic-embed-text',
            'mxbai-embed-large' // Largest
          ];

          let selectedEmbeddingModel = null;

          // Try to find a suitable embedding model that's available
          for (const model of embeddingModels) {
            if (data.models.some(m => m.name === model || m.name.startsWith(model))) {
              selectedEmbeddingModel = data.models.find(
                m => m.name === model || m.name.startsWith(model)
              ).name;
              break;
            }
          }

          // If no embedding-specific model is found, check if any model has "embed" in its name
          if (!selectedEmbeddingModel) {
            const embedModel = data.models.find(m =>
              m.name.toLowerCase().includes('embed') ||
              m.name.toLowerCase().includes('mini')
            );

            if (embedModel) {
              selectedEmbeddingModel = embedModel.name;
            }
          }

          if (selectedEmbeddingModel) {
            setEmbeddingsModel(selectedEmbeddingModel);
            setModelStatus(`Using ${selectedEmbeddingModel} for embeddings`);
          } else {
            setModelStatus('No embedding models found');
            setSearchType('keyword');
            setError('No embedding models available. Using keyword search instead.');
          }

          // Set LLM model if available - prioritize smaller models for reliability
          const llmModels = ['phi2', 'tinyllama', 'mistral-tiny', 'gemma:2b', 'mistral', 'llama2', 'phi3', 'gemma', 'qwen', 'falcon'];
          for (const model of llmModels) {
            if (data.models.some(m => m.name === model || m.name.startsWith(model))) {
              setSelectedModel(data.models.find(
                m => m.name === model || m.name.startsWith(model)
              ).name);
              break;
            }
          }
        } else {
          setModelStatus('No models found');
          setError('No models available. Please pull models using Ollama CLI: ollama pull nomic-embed-text');
        }
      } else {
        setModelStatus('Ollama server error');
        setError('Failed to connect to Ollama server. Please ensure Ollama is running on port 11434.');
      }
    } catch (error) {
      setModelStatus('Ollama server not reachable');
      setError('Cannot reach Ollama server. Please install and start Ollama on your machine.');
    }

    // Try to get system info to estimate available memory
    try {
      const systemResponse = await fetch('http://localhost:11434/api/version');
      if (systemResponse.ok) {
        // If we can't get exact memory limits, set a conservative limit
        setMemoryLimit(4);
      }
    } catch (error) {
      // If we can't get info, assume a conservative limit
      setMemoryLimit(4);
    }
  };

  // Text chunking function for large documents
  const chunkText = (text, size = chunkSize, overlap = chunkOverlap) => {
    // Validate inputs
    if (!text || typeof text !== 'string') return [];
    if (size <= 0) return [text];
    if (overlap >= size) overlap = Math.floor(size / 2);

    const chunks = [];
    let i = 0;

    // If text is smaller than chunk size, return the whole text
    if (text.length <= size) return [text];

    while (i < text.length) {
      // Ensure endIndex doesn't exceed text length
      let endIndex = Math.min(i + size, text.length);

      // Try to find a natural break point if not at the end
      if (endIndex < text.length) {
        // Try to find a natural break point
        const breakChars = ['.', '!', '?', '\n', ' '];
        let foundBreak = false;

        // Look for break characters in the last 20% of the chunk
        const lookaheadSize = Math.min(Math.floor(size * 0.2), 100); // Limit lookahead size
        const minEndIndex = Math.max(i, endIndex - lookaheadSize);

        // Search backwards from endIndex to find a natural break
        for (let j = endIndex; j > minEndIndex; j--) {
          if (breakChars.includes(text[j])) {
            endIndex = j + 1;
            foundBreak = true;
            break;
          }
        }

        // If no break found, look ahead a bit (but don't exceed text length)
        if (!foundBreak) {
          const maxLookahead = Math.min(endIndex + lookaheadSize, text.length);
          for (let j = endIndex; j < maxLookahead; j++) {
            if (breakChars.includes(text[j])) {
              endIndex = j + 1;
              break;
            }
          }
        }
      }

      // Safety check to ensure we're making progress
      if (endIndex <= i) {
        endIndex = Math.min(i + 1, text.length); // Move at least one character forward
      }

      // Add the chunk
      const chunk = text.slice(i, endIndex).trim();
      if (chunk) chunks.push(chunk);

      // Move the start position, accounting for overlap
      i = endIndex - overlap;

      // Ensure we're making progress
      if (i <= 0 || i >= text.length || i === endIndex) {
        i = endIndex; // Skip overlap if it would cause issues
      }
    }

    return chunks;
  };

  // Generate embeddings for a text using the Ollama API
  const generateEmbedding = async (text) => {
    if (!embeddingsModel) return null;

    try {
      console.log(`Generating embedding with model: ${embeddingsModel}`);

      // First, check if the model supports the embeddings endpoint
      try {
        const response = await fetch('http://localhost:11434/api/embeddings', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            model: embeddingsModel,
            prompt: text,
          }),
        });

        if (response.ok) {
          const data = await response.json();
          return data.embedding;
        } else {
          const errorData = await response.text();
          console.log('Embeddings endpoint failed:', errorData);

          // Check if it's a memory error
          if (errorData.includes('system memory') || errorData.includes('out of memory')) {
            // Try to use a smaller model next time
            if (availableModels.length > 0) {
              const currentIndex = availableModels.findIndex(m => m === embeddingsModel);

              // Find a potentially smaller model
              for (let i = 0; i < availableModels.length; i++) {
                const model = availableModels[i];
                if (
                  model.includes('small') ||
                  model.includes('mini') ||
                  (currentIndex !== -1 && i < currentIndex)
                ) {
                  setEmbeddingsModel(model);
                  setError(`Switched to smaller model: ${model} due to memory limitations`);
                  break;
                }
              }
            }
          }
        }
      } catch (error) {
        console.log('Embeddings endpoint error:', error);
      }

      // Fallback to generate endpoint with embedding option
      try {
        const response = await fetch('http://localhost:11434/api/generate', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            model: embeddingsModel,
            prompt: text,
            options: {
              embedding: true,
            },
          }),
        });

        if (!response.ok) {
          const errorData = await response.text();
          console.error('Generate endpoint error:', errorData);

          // Check if it's a memory error
          if (errorData.includes('system memory') || errorData.includes('out of memory')) {
            setSearchType('keyword');
            setError('Not enough memory for embeddings. Switched to keyword search.');
          }

          // Fall back to keyword search
          return null;
        }

        const data = await response.json();
        return data.embedding;
      } catch (error) {
        console.error('Error with generate endpoint:', error);
        return null;
      }
    } catch (error) {
      console.error('Error generating embeddings:', error);
      return null;
    }
  };

  // Enhanced fallback method for keyword-based search
  const createKeywordVector = (text) => {
    if (!text) return {};

    // Extract important keywords and their frequencies
    const words = text.toLowerCase()
      .replace(/[^\w\s]/g, ' ')  // Replace punctuation with spaces
      .split(/\s+/)              // Split on whitespace
      .filter(word => word.length > 2 && !commonStopWords.includes(word));

    // Count word frequencies
    const wordFreq = {};
    words.forEach(word => {
      wordFreq[word] = (wordFreq[word] || 0) + 1;
    });

    return wordFreq;
  };

  // Common English stop words to filter out
  const commonStopWords = [
    'the', 'and', 'that', 'have', 'for', 'not', 'with', 'you', 'this', 'but',
    'his', 'her', 'she', 'he', 'they', 'them', 'their', 'it', 'is', 'was',
    'be', 'been', 'being', 'are', 'were', 'will', 'would', 'should', 'can',
    'could', 'may', 'might', 'must', 'shall', 'from', 'when', 'where', 'how',
    'all', 'any', 'both', 'each', 'few', 'more', 'most', 'some', 'such', 'than'
  ];

  // Calculate cosine similarity between two vectors
  const cosineSimilarity = (vecA, vecB) => {
    // For numeric embedding vectors
    if (Array.isArray(vecA) && Array.isArray(vecB)) {
      if (vecA.length !== vecB.length) return 0;

      let dotProduct = 0;
      let normA = 0;
      let normB = 0;

      for (let i = 0; i < vecA.length; i++) {
        dotProduct += vecA[i] * vecB[i];
        normA += vecA[i] * vecA[i];
        normB += vecB[i] * vecB[i];
      }

      normA = Math.sqrt(normA);
      normB = Math.sqrt(normB);

      if (normA === 0 || normB === 0) return 0;

      return dotProduct / (normA * normB);
    }
    // For keyword-based vectors (objects with word frequencies)
    else if (typeof vecA === 'object' && typeof vecB === 'object') {
      // Get all unique words from both objects
      const allWords = [...new Set([...Object.keys(vecA), ...Object.keys(vecB)])];

      let dotProduct = 0;
      let normA = 0;
      let normB = 0;

      // Calculate dot product and magnitudes
      allWords.forEach(word => {
        const freqA = vecA[word] || 0;
        const freqB = vecB[word] || 0;

        dotProduct += freqA * freqB;
        normA += freqA * freqA;
        normB += freqB * freqB;
      });

      normA = Math.sqrt(normA);
      normB = Math.sqrt(normB);

      if (normA === 0 || normB === 0) return 0;

      return dotProduct / (normA * normB);
    }

    return 0;
  };

  // Handle file upload and process documents
  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    // Check file type first
    const validTypes = ['text/plain', 'text/csv'];
    const validExtensions = ['.txt', '.csv'];
    if (!validTypes.includes(file.type) &&
      !validExtensions.some(ext => file.name.toLowerCase().endsWith(ext))) {
      setError('Invalid file type. Please upload .txt or .csv files.');
      return;
    }

    // Check if Ollama is available before proceeding
    if (modelStatus === 'Ollama server not reachable' || modelStatus === 'Ollama server error') {
      setError('Cannot process documents: Ollama server is not available.');
      return;
    }

    if (file.size > 10 * 1024 * 1024) { // 10MB limit
      setError('File is too large. Please upload files smaller than 10MB.');
      return;
    }

    setLoading(true);
    setError('');

    try {
      // Process based on file type
      if (file.type === 'text/plain' || file.name.endsWith('.txt')) {
        // Process text file
        const text = await file.text();

        // Check if text is empty
        if (!text || text.trim().length === 0) {
          setError('The file is empty or contains only whitespace.');
          setLoading(false);
          return;
        }

        let docs = [];

        if (useChunking) {
          // Split into chunks for better processing
          const chunks = chunkText(text);
          // Validate chunks
          if (!chunks || chunks.length === 0) {
            setError('Failed to chunk document. Please try with chunking disabled.');
            setLoading(false);
            return;
          }
          docs = chunks.map((chunk, index) => ({
            id: `${file.name}-${index}`,
            content: chunk,
            embedding: null,
          }));
        } else {
          // Split by paragraphs if not using automatic chunking
          const paragraphs = text.split('\n\n')
            .filter(paragraph => paragraph.trim().length > 0);

          if (paragraphs.length === 0) {
            setError('No valid content found in the document.');
            setLoading(false);
            return;
          }

          docs = paragraphs.map((paragraph, index) => ({
            id: `${file.name}-${index}`,
            content: paragraph,
            embedding: null,
          }));
        }

        await processDocs(docs);
      } else if (file.type === 'text/csv' || file.name.endsWith('.csv')) {
        // Process CSV file
        Papaparse.parse(file, {
          header: true,
          skipEmptyLines: true,
          dynamicTyping: true, // Automatically convert numeric values
          delimitersToGuess: [',', '\t', '|', ';'],
          complete: async (results) => {
            // Check if parsing was successful
            if (results.errors && results.errors.length > 0) {
              setError(`CSV parsing error: ${results.errors[0].message}`);
              setLoading(false);
              return;
            }

            // Make sure we have data
            if (!results.data || results.data.length === 0) {
              setError('The CSV file appears to be empty.');
              setLoading(false);
              return;
            }

            // Determine the column with text content (use the first non-ID column)
            const columns = Object.keys(results.data[0] || {});
            const contentColumn = columns.find(col =>
              col.toLowerCase() !== 'id' &&
              typeof results.data[0][col] === 'string'
            ) || columns[0];

            let docs = results.data
              .filter(row => row[contentColumn]?.toString().trim().length > 0)
              .map((row, index) => ({
                id: row.id || index,
                content: row[contentColumn]?.toString(),
                embedding: null,
                metadata: { ...row } // Store all columns as metadata
              }));

            if (useChunking) {
              // If the content is large, chunk it further
              const newDocs = [];
              docs.forEach(doc => {
                if (doc.content && doc.content.length > chunkSize) {
                  const chunks = chunkText(doc.content);
                  chunks.forEach((chunk, i) => {
                    newDocs.push({
                      id: `${doc.id}-chunk-${i}`,
                      content: chunk,
                      embedding: null,
                      metadata: doc.metadata,
                      originalId: doc.id
                    });
                  });
                } else {
                  newDocs.push(doc);
                }
              });
              docs = newDocs;
            }

            processDocs(docs);
          },
          error: (error) => {
            setError(`Error parsing CSV: ${error.message}`);
            setLoading(false);
          }
        });
      } else {
        setError('Unsupported file type. Please upload a .txt or .csv file.');
        setLoading(false);
      }
    } catch (error) {
      console.error('File processing error:', error);
      setError(`Error processing file: ${error.message || 'Invalid file format'}`);
      setLoading(false);
    } finally {
      event.target.value = ''; // Reset file input
    }
  };

  // Process documents and generate embeddings for them
  const processDocs = async (docs) => {
    setLoading(true);
    try {
      // Check if we can use semantic search
      const canUseEmbeddings = embeddingsModel && searchType === 'semantic';
      let docsWithEmbeddings = [];

      if (canUseEmbeddings) {
        // Process in smaller batches to avoid overwhelming the API
        const batchSize = 3; // Smaller batch size to reduce memory pressure
        for (let i = 0; i < docs.length; i += batchSize) {
          const batch = docs.slice(i, i + batchSize);

          // Process batch concurrently
          const batchResults = await Promise.all(
            batch.map(async (doc) => {
              try {
                // For longer documents, generate embedding or keyword vector
                const embedding = await generateEmbedding(doc.content);
                const keywordVector = createKeywordVector(doc.content);

                return {
                  ...doc,
                  embedding,
                  keywordVector
                };
              } catch (error) {
                console.error(`Error processing document ${doc.id}:`, error);
                return {
                  ...doc,
                  embedding: null,
                  keywordVector: createKeywordVector(doc.content)
                };
              }
            })
          );

          docsWithEmbeddings = [...docsWithEmbeddings, ...batchResults];

          // Check if embeddings are failing consistently
          const firstBatchFailed = i === 0 && batchResults.every(doc => doc.embedding === null);
          if (firstBatchFailed) {
            // Switch to keyword search early to avoid wasting time
            setSearchType('keyword');
            setError('Embedding generation failed. Using keyword search instead.');

            // Process the rest of the documents with only keyword vectors
            const remainingDocs = docs.slice(i + batchSize);
            const remainingResults = remainingDocs.map(doc => ({
              ...doc,
              embedding: null,
              keywordVector: createKeywordVector(doc.content)
            }));

            docsWithEmbeddings = [...docsWithEmbeddings, ...remainingResults];
            break;
          }
        }
      } else {
        // If embeddings aren't available, just use keyword vectors
        docsWithEmbeddings = docs.map(doc => ({
          ...doc,
          embedding: null,
          keywordVector: createKeywordVector(doc.content)
        }));
      }

      setDocuments(docsWithEmbeddings);
      setError('');

      // If all embeddings failed, switch to keyword search
      const allEmbeddingsFailed = docsWithEmbeddings.every(doc => doc.embedding === null);
      if (allEmbeddingsFailed && searchType === 'semantic') {
        setSearchType('keyword');
        setError('Failed to generate embeddings. Automatically switched to keyword search.');
      }
    } catch (error) {
      setError('Error processing documents: ' + error.message);
      setSearchType('keyword'); // Fallback to keyword search
    } finally {
      setLoading(false);
    }
  };

  // Search for documents based on the query
  const handleSearch = async () => {
    if (!searchQuery.trim()) return;

    setLoading(true);
    setError('');
    try {
      if (searchType === 'semantic' && embeddingsModel) {
        // Semantic search using embeddings
        const queryEmbedding = await generateEmbedding(searchQuery);

        if (!queryEmbedding) {
          // Fallback to keyword search if embeddings fail
          setError('Failed to generate embedding for query. Using keyword search instead.');
          performKeywordSearch();
          return;
        }

        // Calculate similarity and sort by relevance
        const results = documents
          .filter(doc => doc.embedding !== null) // Only consider docs with embeddings
          .map(doc => ({
            ...doc,
            similarity: cosineSimilarity(queryEmbedding, doc.embedding)
          }))
          .sort((a, b) => b.similarity - a.similarity)
          .slice(0, 5); // Top 5 results

        // Only include results with reasonable similarity
        const filteredResults = results.filter(doc => doc.similarity > 0.2);

        if (filteredResults.length > 0) {
          setSearchResults(filteredResults);
        } else {
          // If no good results with embeddings, fallback to keyword search
          setError('No semantically similar results found. Using keyword search as fallback.');
          performKeywordSearch();
        }
      } else {
        // Perform keyword search
        performKeywordSearch();
      }
    } catch (error) {
      setError('Search error: ' + error.message);
      performKeywordSearch(); // Fallback
    } finally {
      setLoading(false);
    }
  };

  // Enhanced keyword search using TF-IDF like approach
  const performKeywordSearch = () => {
    // Create a keyword vector for the query
    const queryVector = createKeywordVector(searchQuery);

    // Check if we have any keywords in the query
    if (Object.keys(queryVector).length === 0) {
      setSearchResults([]);
      setError('Search query too short or contains only common words. Please try a more specific query.');
      return;
    }

    // For basic keyword matching
    const normalizedQuery = searchQuery.toLowerCase();
    const queryTerms = normalizedQuery.split(/\s+/).filter(term => term.length > 2);

    // Calculate similarity scores
    let results = documents.map(doc => {
      // Use keyword vectors for similarity if available
      let similarity = 0;

      if (doc.keywordVector) {
        similarity = cosineSimilarity(queryVector, doc.keywordVector);
      }

      // Boost score for exact matches
      const normalizedContent = doc.content.toLowerCase();
      queryTerms.forEach(term => {
        if (normalizedContent.includes(term)) {
          similarity += 0.1;
        }
      });

      // Higher boost for exact phrase match
      if (normalizedContent.includes(normalizedQuery)) {
        similarity += 0.3;
      }

      return {
        ...doc,
        similarity
      };
    });

    // Filter out documents with no match
    results = results
      .filter(doc => doc.similarity > 0)
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, 5);

    if (results.length === 0) {
      setError('No matching documents found for your query.');
    }

    setSearchResults(results);
  };

  // Generate answer using Ollama with error handling and fallbacks
  const generateAnswer = async () => {
    if (!question.trim() || searchResults.length === 0) return;

    setLoading(true);
    setError('');
    setAnswer('');

    try {
      // Make sure we have a valid model
      let modelToUse = selectedModel;
      let fallbackAttempted = false;

      // If the model isn't available or is too large, find a fallback
      const modelPriorityList = [
        'phi2',        // ~2GB
        'tinyllama',   // ~2GB
        'mistral-tiny',// ~2GB
        'gemma:2b',    // ~2GB
        'mistral',     // ~4GB
        'llama2',      // ~4GB
        'phi3',        // ~4GB
        'gemma',       // ~5GB
        'qwen',        // ~5GB
        'falcon'       // ~6GB
      ];

      // Try up to 3 times with progressively smaller models
      for (let attempt = 0; attempt < 3; attempt++) {
        try {
          // On subsequent attempts, try smaller models
          if (attempt > 0) {
            const availableSmallerModels = modelPriorityList.filter(model =>
              availableModels.some(m => m === model || m.startsWith(model))
            );

            if (availableSmallerModels.length > 0) {
              modelToUse = availableSmallerModels[0];
              fallbackAttempted = true;
              setError(`Memory issue detected. Trying smaller model: ${modelToUse}`);
            }
          }

          // Prepare context from search results - limit size to prevent memory issues
          const maxContextLength = 2000; // Limit context to prevent memory overload
          const context = searchResults
            .slice(0, 3) // Only use top 3 results
            .map(doc => doc.content.length > maxContextLength
              ? doc.content.substring(0, maxContextLength) + '...'
              : doc.content
            )
            .join('\n\n');

          // Improved prompt template with strict instructions
          const prompt = `Answer the question based ONLY on the following context. 
        If you don't know the answer, say "I don't know".
        
        Context:
        ${context}
        
        Question: ${question}
        
        Answer:`;

          const response = await fetch('http://localhost:11434/api/generate', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              model: modelToUse,
              prompt: prompt,
              stream: false,
              options: {
                num_predict: 200,  // Reduced from 500 to save memory
                temperature: 0.3,
                top_k: 40,
                top_p: 0.9
              }
            }),
          });

          if (!response.ok) {
            const errorText = await response.text();
            throw new Error(errorText);
          }

          const data = await response.json();
          setAnswer(data.response);

          // If we got here successfully, break out of the retry loop
          if (fallbackAttempted) {
            setSelectedModel(modelToUse); // Remember the working model
          }
          return;
        } catch (error) {
          console.error(`Attempt ${attempt + 1} failed:`, error);

          // If this was our last attempt, throw the error
          if (attempt === 2) {
            throw error;
          }

          // Wait a bit before retrying
          await new Promise(resolve => setTimeout(resolve, 500));
        }
      }
    } catch (error) {
      console.error('Error generating answer:', error);

      // Special handling for memory errors
      if (error.message.includes('system memory')) {
        setError('The model requires more memory than available. Try a smaller model or close other applications.');
      } else {
        setError('Failed to generate answer: ' + error.message);
      }

      setAnswer('');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-6 max-w-6xl mx-auto">
      <h1 className="text-3xl font-bold mb-6">Client-Side RAG with React & Ollama</h1>

      {/* Model Status */}
      <div className="mb-6 p-4 bg-gray-100 text-black rounded-lg">
        <h2 className="text-xl font-semibold mb-2">Model Status</h2>
        <div className="flex flex-col gap-2">
          <p><span className="font-medium">Ollama Status:</span> {modelStatus}</p>
          <div className="flex gap-2 items-center">
            <span className="font-medium">LLM Model:</span>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="border rounded px-2 py-1"
            >
              {availableModels.length > 0 ? (
                availableModels.map(model => {
                  // Add estimated memory requirements based on model name
                  let memoryEstimate = '';
                  if (model.includes('phi2') || model.includes('tinyllama') || model.includes('2b')) {
                    memoryEstimate = ' (~2GB)';
                  } else if (model.includes('mistral') || model.includes('llama2') || model.includes('phi3')) {
                    memoryEstimate = ' (~4GB)';
                  } else if (model.includes('gemma') || model.includes('qwen')) {
                    memoryEstimate = ' (~5GB)';
                  } else if (model.includes('falcon')) {
                    memoryEstimate = ' (~6GB)';
                  } else if (model.includes('7b') || model.includes('13b')) {
                    memoryEstimate = ' (~8GB+)';
                  }

                  return (
                    <option key={model} value={model}>
                      {model}{memoryEstimate}
                    </option>
                  );
                })
              ) : (
                <>
                  <option value="phi2">Phi-2 (~2GB)</option>
                  <option value="tinyllama">TinyLlama (~2GB)</option>
                  <option value="mistral">Mistral (~4GB)</option>
                  <option value="llama2">Llama 2 (~4GB)</option>
                </>
              )}
            </select>
          </div>
          <div className="flex gap-2 items-center">
            <span className="font-medium">Search Type:</span>
            <select
              value={searchType}
              onChange={(e) => setSearchType(e.target.value)}
              className="border rounded px-2 py-1"
            >
              <option value="semantic">Semantic Search</option>
              <option value="keyword">Keyword Search</option>
            </select>
          </div>
          <div className="mt-2">
            <p className="text-sm">
              {embeddingsModel ? (
                <>Using <span className="font-medium">{embeddingsModel}</span> for embeddings</>
              ) : (
                <span className="text-red-600">No embedding model available</span>
              )}
            </p>
          </div>
        </div>
      </div>

      {/* Memory Management Options */}
      <div className="mb-6 p-4 bg-blue-50 text-black rounded-lg">
        <h2 className="text-xl font-semibold mb-2">Memory Management</h2>
        <div className="flex flex-col gap-2">
          <div className="flex items-center gap-2">
            <input
              type="checkbox"
              id="chunking"
              checked={useChunking}
              onChange={(e) => setUseChunking(e.target.checked)}
            />
            <label htmlFor="chunking" className="font-medium">Enable Document Chunking</label>
          </div>

          {useChunking && (
            <div className="ml-5 flex flex-col gap-2">
              <div className="flex items-center gap-2">
                <label htmlFor="chunkSize" className="w-24">Chunk Size:</label>
                <input
                  type="number"
                  id="chunkSize"
                  value={chunkSize}
                  onChange={(e) => setChunkSize(parseInt(e.target.value) || 500)}
                  className="border rounded px-2 py-1 w-24"
                  min="100"
                  max="2000"
                />
                <span className="text-sm text-gray-500">characters</span>
              </div>

              <div className="flex items-center gap-2">
                <label htmlFor="chunkOverlap" className="w-24">Overlap:</label>
                <input
                  type="number"
                  id="chunkOverlap"
                  value={chunkOverlap}
                  onChange={(e) => setChunkOverlap(parseInt(e.target.value) || 50)}
                  className="border rounded px-2 py-1 w-24"
                  min="0"
                  max="200"
                />
                <span className="text-sm text-gray-500">characters</span>
              </div>
            </div>
          )}

          <div className="mt-2 text-sm text-gray-700">
            <p>💡 <strong>Tips for reducing memory usage:</strong></p>
            <ul className="list-disc ml-5 mt-1">
              <li>Use smaller embedding models (all-minilm, e5-small)</li>
              <li>Enable chunking for large documents</li>
              <li>Switch to keyword search if experiencing memory errors</li>
              <li>Try installing Ollama models with quantization (e.g. <code>ollama pull all-minilm:1.5b-q4_0</code>)</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <div className="mb-6 p-4 bg-red-100 text-red-700 rounded-lg">
          {error}
        </div>
      )}

      {/* Document Upload */}
      <div className="mb-6 p-4 border rounded-lg">
        <h2 className="text-xl font-semibold mb-2">1. Upload Documents</h2>
        <p className="mb-2">Upload text files or CSV files to create your knowledge base.</p>
        <div className="flex items-center gap-2">
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileUpload}
            accept=".txt,.csv"
            className="hidden"
          />
          <button
            onClick={() => fileInputRef.current?.click()}
            className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
            disabled={loading || modelLoading}
          >
            Upload File
          </button>
          <span>{documents.length > 0 ? `${documents.length} documents loaded` : 'No documents loaded'}</span>
        </div>
      </div>

      {/* Search */}
      <div className="mb-6 p-4 border rounded-lg">
        <h2 className="text-xl font-semibold mb-2">2. Search Knowledge Base</h2>
        <div className="flex gap-2 mb-4">
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Enter search query..."
            className="flex-1 px-3 py-2 border rounded"
            disabled={documents.length === 0 || loading}
          />
          <button
            onClick={handleSearch}
            className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700"
            disabled={documents.length === 0 || !searchQuery.trim() || loading}
          >
            Search
          </button>
        </div>

        {/* Search Results */}
        <div className="mt-4">
          <h3 className="text-lg font-medium mb-2">Search Results:</h3>
          {searchResults.length > 0 ? (
            <div className="space-y-2">
              {searchResults.map((doc) => (
                <div key={doc.id} className="p-3 bg-gray-50 text-black rounded border">
                  {doc.similarity !== undefined && (
                    <div className="text-sm text-gray-500 mb-1">
                      Relevance: {(doc.similarity * 100).toFixed(1) + '%'}
                    </div>
                  )}
                  <p className="text-sm"><strong>Document ID:</strong> {doc.id}</p>
                  <p>{doc.content}</p>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-gray-500">No results to display</p>
          )}
        </div>
      </div>

      {/* Question Answering */}
      <div className="p-4 border rounded-lg">
        <h2 className="text-xl font-semibold mb-2">3. Ask Questions</h2>
        <div className="mb-4">
          <input
            type="text"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Ask a question about the documents..."
            className="w-full px-3 py-2 border rounded mb-2"
            disabled={searchResults.length === 0 || loading}
          />
          <button
            onClick={generateAnswer}
            className="px-4 py-2 bg-purple-600 text-white rounded hover:bg-purple-700"
            disabled={searchResults.length === 0 || !question.trim() || loading}
          >
            Generate Answer
          </button>
        </div>

        {/* Answer */}
        {answer && (
          <div className="p-4 bg-blue-50 text-black rounded border">
            <h3 className="text-lg font-medium mb-2">Answer:</h3>
            <p className="whitespace-pre-wrap">{answer}</p>
          </div>
        )}
      </div>

      {/* Troubleshooting Guide */}
      <div className="mt-6 p-4 bg-yellow-50 border-l-4 border-yellow-400 text-black rounded-lg">
        <h2 className="text-xl font-semibold mb-2">Troubleshooting</h2>
        <div className="text-sm">
          <p className="mb-2">If you're encountering memory errors, try these solutions:</p>
          <ol className="list-decimal ml-5 space-y-1">
            <li>Install smaller embedding models: <code>ollama pull all-minilm:1.5b</code></li>
            <li>Use quantized models: <code>ollama pull nomic-embed-text:q4_0</code></li>
            <li>Increase your system's swap space or available memory</li>
            <li>Switch to keyword search using the dropdown above</li>
            <li>Process smaller documents or reduce chunk size</li>
            <li>Close other memory-intensive applications while using this app</li>
          </ol>
        </div>
      </div>
    </div>
  );
}

export default App;