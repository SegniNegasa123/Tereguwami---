import SwiftUI
import WebKit

struct ContentView: View {
    var body: some View {
        TereguwamiWebView(urlString: "http://127.0.0.1:8000/app")
            .edgesIgnoringSafeArea(.all)
    }
}

struct TereguwamiWebView: UIViewRepresentable {
    let urlString: String

    func makeUIView(context: Context) -> WKWebView {
        let preferences = WKPreferences()
        preferences.javaScriptCanOpenWindowsAutomatically = true

        let config = WKWebViewConfiguration()
        config.preferences = preferences
        config.allowsInlineMediaPlayback = true

        let webView = WKWebView(frame: .zero, configuration: config)
        if let url = URL(string: urlString) {
            webView.load(URLRequest(url: url))
        }
        return webView
    }

    func updateUIView(_ uiView: WKWebView, context: Context) {}
}
