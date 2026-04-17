import SwiftUI
import MLX
import MLXLLM

struct Message: Identifiable {
    let id = UUID()
    var role: String
    var content: String
}

@MainActor
class ChatViewModel: ObservableObject {
    @Published var messages: [Message] = []
    @Published var inputText: String = ""
    @Published var isGenerating = false
    @Published var isDownloading = false
    @Published var downloadProgress: Double = 0
    @Published var statusMessage: String = "Iniciando..."
    
    private var modelContainer: ModelContainer?

    init() {
        Task { await setup() }
    }

    func setup() async {
        isDownloading = true
        statusMessage = "Descargando modelo Qwen3.5..."
        
        do {
            // Descarga automática
            let modelPath = try await downloadModel()
            
            statusMessage = "Cargando modelo..."
            modelContainer = try await ModelContainer.loadModel(at: modelPath)
            
            isDownloading = false
            messages.append(Message(role: "assistant", content: "¡Hola! Soy Qwen3.5. ¿En qué puedo ayudarte?"))
        } catch {
            isDownloading = false
            messages.append(Message(role: "assistant", content: "Error: \(error.localizedDescription)"))
        }
    }

    func downloadModel() async throws -> String {
        let cacheDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
        let modelDir = cacheDir.appendingPathComponent("models").appendingPathComponent("qwen3.5-9b")
        
        if FileManager.default.fileExists(atPath: modelDir.path) {
            return modelDir.path
        }
        
        try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)
        
        // Usa la API de HuggingFace directamente
        let modelId = "mlx-community/Qwen3.5-9B-Instruct-4bit"
        let baseURL = "https://huggingface.co/\(modelId)/resolve/main"
        
        // Archivos del modelo
        let modelFiles = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"]
        
        for (index, file) in modelFiles.enumerated() {
            statusMessage = "Descargando \(file)... (\(index + 1)/\(modelFiles.count))"
            let url = URL(string: "\(baseURL)/\(file)")!
            let destURL = modelDir.appendingPathComponent(file)
            
            let (data, _) = try await URLSession.shared.download(from: url)
            try FileManager.default.moveItem(at: data, to: destURL)
            downloadProgress = Double(index + 1) / Double(modelFiles.count)
        }
        
        return modelDir.path
    }

    func sendMessage() {
        guard !inputText.isEmpty else { return }
        
        let userMessage = Message(role: "user", content: inputText)
        messages.append(userMessage)
        let input = inputText
        inputText = ""
        isGenerating = true

        Task {
            do {
                let response = try await generateResponse(prompt: input)
                messages.append(Message(role: "assistant", content: response))
            } catch {
                messages.append(Message(role: "assistant", content: "Error: \(error.localizedDescription)"))
            }
            isGenerating = false
        }
    }

    func generateResponse(prompt: String) async throws -> String {
        guard let model = modelContainer else { return "Modelo no cargado" }
        
        let fullPrompt = """
        <|im_start|>user
        \(prompt)
        <|im_end|>
        <|im_start|>assistant
        """
        
        var response = ""
        let params = GenerateParameters(temperature: 0.7, maxTokens: 2048)
        
        for try await token in try await model.stream(prompt: fullPrompt, parameters: params) {
            response += token
        }
        
        return response.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

struct ContentView: View {
    @StateObject private var viewModel = ChatViewModel()

    var body: some View {
        VStack(spacing: 0) {
            if viewModel.isDownloading {
                downloadingView
            } else {
                chatView
            }
        }
    }
    
    var downloadingView: some View {
        VStack(spacing: 20) {
            ProgressView()
                .scaleEffect(2)
            Text(viewModel.statusMessage)
                .font(.headline)
            ProgressView(value: viewModel.downloadProgress)
                .frame(width: 200)
            Text("\(Int(viewModel.downloadProgress * 100))%")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
    
    var chatView: some View {
        VStack(spacing: 0) {
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(spacing: 12) {
                        ForEach(viewModel.messages) { message in
                            ChatBubble(message: message)
                        }
                    }
                    .padding()
                }
                .onChange(of: viewModel.messages.count) { _, _ in
                    if let last = viewModel.messages.last {
                        proxy.scrollTo(last.id, anchor: .bottom)
                    }
                }
            }
            
            HStack(spacing: 12) {
                TextField("Escribe tu mensaje...", text: $viewModel.inputText, axis: .vertical)
                    .textFieldStyle(.plain)
                    .padding(12)
                    .background(Color(.secondarySystemBackground))
                    .cornerRadius(20)
                    .lineLimit(1...5)
                    .disabled(viewModel.isGenerating)
                
                Button { viewModel.sendMessage() } label: {
                    Image(systemName: "arrow.up.circle.fill")
                        .font(.system(size: 32))
                        .foregroundStyle(viewModel.inputText.isEmpty ? .gray : .blue)
                }
                .disabled(viewModel.isGenerating || viewModel.inputText.isEmpty)
            }
            .padding()
        }
    }
}

struct ChatBubble: View {
    let message: Message
    
    var body: some View {
        HStack {
            if message.role == "user" { Spacer(minLength: 60) }
            Text(message.content)
                .padding(.horizontal, 16)
                .padding(.vertical, 10)
                .background(message.role == "user" ? Color.blue : Color(.secondarySystemBackground))
                .foregroundColor(message.role == "user" ? .white : .primary)
                .cornerRadius(18)
            if message.role == "assistant" { Spacer(minLength: 60) }
        }
    }
}

