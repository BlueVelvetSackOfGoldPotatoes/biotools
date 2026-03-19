#include "turing/analysis.hpp"
#include "turing/json.hpp"
#include "turing/model.hpp"
#include "turing/simulator.hpp"

#include <boost/asio.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <thread>

namespace beast = boost::beast;
namespace http = beast::http;
namespace net = boost::asio;
using tcp = net::ip::tcp;

namespace {

std::string mime_type(const std::string& path) {
    if (path.ends_with(".html")) {
        return "text/html; charset=utf-8";
    }
    if (path.ends_with(".js")) {
        return "text/javascript; charset=utf-8";
    }
    if (path.ends_with(".css")) {
        return "text/css; charset=utf-8";
    }
    if (path.ends_with(".json")) {
        return "application/json; charset=utf-8";
    }
    if (path.ends_with(".svg")) {
        return "image/svg+xml";
    }
    return "text/plain; charset=utf-8";
}

std::string read_text_file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

std::filesystem::path default_static_root_for_executable(const std::filesystem::path& executable) {
    const auto executable_dir = std::filesystem::absolute(executable).parent_path();
    const auto from_executable = executable_dir / ".." / "frontend" / "dist";
    if (std::filesystem::exists(from_executable)) {
        return from_executable;
    }

    const auto from_cwd = std::filesystem::current_path() / "frontend" / "dist";
    if (std::filesystem::exists(from_cwd)) {
        return from_cwd;
    }

    return from_cwd;
}

http::response<http::string_body> make_json_response(
    http::status status,
    const boost::json::value& payload) {
    http::response<http::string_body> response{status, 11};
    response.set(http::field::content_type, "application/json; charset=utf-8");
    response.body() = boost::json::serialize(payload);
    response.prepare_payload();
    return response;
}

http::response<http::string_body> make_error_response(
    http::status status,
    const std::string& message) {
    boost::json::object payload;
    payload["error"] = message;
    return make_json_response(status, payload);
}

http::response<http::string_body> serve_static_file(
    const http::request<http::string_body>& request,
    const std::filesystem::path& static_root) {
    auto relative = std::string(request.target());
    if (relative.empty() || relative == "/") {
        relative = "/index.html";
    }

    const auto clean_relative = relative[0] == '/' ? relative.substr(1) : relative;
    auto path = static_root / clean_relative;
    if (!std::filesystem::exists(path)) {
        path = static_root / "index.html";
    }
    if (!std::filesystem::exists(path)) {
        http::response<http::string_body> response{http::status::not_found, request.version()};
        response.set(http::field::content_type, "text/plain; charset=utf-8");
        response.body() =
            "Frontend build not found. Run the React dev server in frontend/ or build frontend/dist.";
        response.prepare_payload();
        return response;
    }

    http::response<http::string_body> response{http::status::ok, request.version()};
    response.set(http::field::content_type, mime_type(path.string()));
    response.body() = read_text_file(path);
    response.prepare_payload();
    return response;
}

class ApiService {
public:
    http::response<http::string_body> handle(
        const http::request<http::string_body>& request) {
        try {
            if (request.method() == http::verb::get && request.target() == "/api/health") {
                boost::json::object payload;
                payload["status"] = "ok";
                return make_json_response(http::status::ok, payload);
            }

            if (request.method() == http::verb::get && request.target() == "/api/presets") {
                boost::json::array presets;
                for (const auto& preset : registry_.presets()) {
                    presets.push_back(turing::to_json(preset));
                }
                boost::json::array families;
                for (const auto& family : registry_.families()) {
                    families.push_back(turing::to_json(family));
                }
                boost::json::object payload;
                payload["presets"] = presets;
                payload["families"] = families;
                return make_json_response(http::status::ok, payload);
            }

            if (request.method() != http::verb::post) {
                return make_error_response(http::status::method_not_allowed, "Unsupported HTTP method.");
            }

            const auto body = request.body().empty() ? boost::json::value(boost::json::object{}) : boost::json::parse(request.body());
            if (!body.is_object()) {
                return make_error_response(http::status::bad_request, "JSON body must be an object.");
            }
            const auto& object = body.as_object();

            if (request.target() == "/api/simulate") {
                const auto config = turing::simulation_config_from_json(object, registry_);
                turing::validate_config(config);
                const auto result = experiment_.simulate(config);
                return make_json_response(http::status::ok, turing::to_json(result));
            }

            if (request.target() == "/api/batch") {
                const auto config = turing::simulation_config_from_json(object, registry_);
                turing::validate_config(config);
                const auto replicates = turing::batch_replicates_from_json(object, 128);
                const auto result = experiment_.simulate_batch(config, replicates);
                return make_json_response(http::status::ok, turing::to_json(result));
            }

            if (request.target() == "/api/analyze") {
                const auto config = turing::simulation_config_from_json(object, registry_);
                turing::validate_config(config);
                auto gamma = config.incipient_capture_gamma;
                if (const auto* value = object.if_contains("gamma")) {
                    if (value->is_double()) {
                        gamma = value->as_double();
                    } else if (value->is_int64()) {
                        gamma = static_cast<double>(value->as_int64());
                    } else if (value->is_uint64()) {
                        gamma = static_cast<double>(value->as_uint64());
                    }
                }
                const auto analysis = analyzer_.analyze(model_, config, gamma);
                return make_json_response(http::status::ok, turing::to_json(analysis));
            }

            return make_error_response(http::status::not_found, "Unknown API endpoint.");
        } catch (const std::exception& exception) {
            return make_error_response(http::status::bad_request, exception.what());
        }
    }

private:
    turing::PresetRegistry registry_;
    turing::PaperModel model_;
    turing::LinearStabilityAnalyzer analyzer_;
    turing::RingExperiment experiment_;
};

void do_session(
    tcp::socket socket,
    const std::filesystem::path& static_root,
    ApiService& service) {
    beast::flat_buffer buffer;
    beast::error_code error_code;

    for (;;) {
        http::request<http::string_body> request;
        http::read(socket, buffer, request, error_code);
        if (error_code == http::error::end_of_stream) {
            break;
        }
        if (error_code) {
            std::cerr << "Read error: " << error_code.message() << '\n';
            break;
        }

        http::response<http::string_body> response;
        if (std::string(request.target()).starts_with("/api/")) {
            response = service.handle(request);
            response.version(request.version());
            response.keep_alive(request.keep_alive());
        } else {
            response = serve_static_file(request, static_root);
            response.version(request.version());
            response.keep_alive(request.keep_alive());
        }

        http::write(socket, response, error_code);
        if (error_code) {
            std::cerr << "Write error: " << error_code.message() << '\n';
            break;
        }
        if (!response.keep_alive()) {
            break;
        }
    }

    socket.shutdown(tcp::socket::shutdown_send, error_code);
}

std::optional<std::string> argument_value(int argc, char* argv[], const std::string& key) {
    for (int index = 1; index + 1 < argc; ++index) {
        if (argv[index] == key) {
            return argv[index + 1];
        }
    }
    return std::nullopt;
}

}  // namespace

int main(int argc, char* argv[]) {
    const auto host = argument_value(argc, argv, "--host").value_or("0.0.0.0");
    const auto port = static_cast<unsigned short>(
        std::stoi(argument_value(argc, argv, "--port").value_or("8080")));
    const auto static_root = std::filesystem::path(
        argument_value(argc, argv, "--static-root").value_or(
            default_static_root_for_executable(argv[0]).string()));

    try {
        net::io_context ioc{1};
        tcp::acceptor acceptor{ioc, {net::ip::make_address(host), port}};
        ApiService service;

        std::cout << "Turing ring backend listening on http://" << host << ':' << port << '\n';
        std::cout << "Serving static files from " << std::filesystem::absolute(static_root) << '\n';

        for (;;) {
            tcp::socket socket{ioc};
            acceptor.accept(socket);
            std::thread(
                [&service, static_root](tcp::socket accepted_socket) mutable {
                    do_session(std::move(accepted_socket), static_root, service);
                },
                std::move(socket))
                .detach();
        }
    } catch (const std::exception& exception) {
        std::cerr << "Fatal error: " << exception.what() << '\n';
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
