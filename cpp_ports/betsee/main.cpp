// ---------------------------------------------------------------------------
// main.cpp - BETSEE C++17/Qt5 Port - Application Entry Point
// BioElectric Tissue Simulation Engine Environment
// Original Python/PySide2 by Alexis Pietak & Cecil Curry.
// ---------------------------------------------------------------------------

#include "betsee.h"

#ifdef HAS_QT5

#include <QApplication>
#include <QCommandLineParser>
#include <QSplashScreen>
#include <QPixmap>
#include <QTimer>
#include <QMessageBox>
#include <QDir>
#include <QStandardPaths>
#include <csignal>
#include <iostream>

// ---------------------------------------------------------------------------
// Global exception handler (port of util/io/guierror.py)
// ---------------------------------------------------------------------------

static void installExceptionHandler() {
    // Install a terminate handler to display errors via Qt message box
    std::set_terminate([]() {
        try {
            auto eptr = std::current_exception();
            if (eptr) std::rethrow_exception(eptr);
        } catch (const std::exception& e) {
            QMessageBox::critical(nullptr,
                QObject::tr("Fatal Error"),
                QObject::tr("An unhandled exception occurred:\n\n%1")
                    .arg(e.what()));
        } catch (...) {
            QMessageBox::critical(nullptr,
                QObject::tr("Fatal Error"),
                QObject::tr("An unknown fatal error occurred."));
        }
        std::abort();
    });
}

// ---------------------------------------------------------------------------
// Signal handler for graceful shutdown
// ---------------------------------------------------------------------------

static betsee::BetseeMainWindow* g_mainWindow = nullptr;

static void signalHandler(int signum) {
    if (g_mainWindow) {
        g_mainWindow->close();
    }
    std::signal(signum, SIG_DFL);
}

// ---------------------------------------------------------------------------
// Application entry point (port of betsee/cli/guicli.py + betsee/__main__.py)
// ---------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    // Set application attributes before creating QApplication
    // (port of util/app/guiapp.py _init_qt())
    QApplication::setAttribute(Qt::AA_EnableHighDpiScaling, true);
    QApplication::setAttribute(Qt::AA_UseHighDpiPixmaps, true);

    // Create the QApplication singleton
    QApplication app(argc, argv);

    // Set application metadata (port of guimetadata.py)
    QApplication::setApplicationName(betsee::AppMetadata::NAME);
    QApplication::setApplicationVersion(betsee::AppMetadata::VERSION);
    QApplication::setOrganizationName(betsee::AppMetadata::ORG_NAME);
    QApplication::setOrganizationDomain(betsee::AppMetadata::ORG_DOMAIN);
    QApplication::setApplicationDisplayName(
        QObject::tr("BETSEE - BioElectric Tissue Simulation Engine Environment"));

    // Install global exception handler
    installExceptionHandler();

    // Install signal handlers for graceful shutdown
    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);

    // Parse command line arguments (port of cli/guicli.py)
    QCommandLineParser parser;
    parser.setApplicationDescription(betsee::AppMetadata::DESCRIPTION);
    parser.addHelpOption();
    parser.addVersionOption();

    QCommandLineOption simConfFileOption(
        QStringList() << "sim-conf-file" << "f",
        QObject::tr("Simulation configuration file to initially open."),
        QObject::tr("file"));
    parser.addOption(simConfFileOption);

    QCommandLineOption verboseOption(
        QStringList() << "verbose" << "v",
        QObject::tr("Enable verbose logging output."));
    parser.addOption(verboseOption);

    QCommandLineOption cachePolicyOption(
        "cache-policy",
        QObject::tr("Type of caching to perform (auto, dev, user)."),
        QObject::tr("policy"),
        "auto");
    parser.addOption(cachePolicyOption);

    // Also accept a positional argument for the config file
    parser.addPositionalArgument("config",
        QObject::tr("Optional simulation configuration file to open."),
        QObject::tr("[config]"));

    parser.process(app);

    // Determine the initial simulation configuration file
    QString simConfFilename;

    // Check the --sim-conf-file option first
    if (parser.isSet(simConfFileOption)) {
        simConfFilename = parser.value(simConfFileOption);
    }
    // Then check positional arguments
    else {
        const QStringList posArgs = parser.positionalArguments();
        if (!posArgs.isEmpty()) {
            simConfFilename = posArgs.first();
        }
    }

    // Validate the config file if specified
    if (!simConfFilename.isEmpty()) {
        QFileInfo fileInfo(simConfFilename);
        if (!fileInfo.exists()) {
            std::cerr << "Error: Simulation configuration file not found: "
                      << simConfFilename.toStdString() << std::endl;
            return 1;
        }
        if (!fileInfo.isReadable()) {
            std::cerr << "Error: Simulation configuration file not readable: "
                      << simConfFilename.toStdString() << std::endl;
            return 1;
        }
        // Convert to absolute path
        simConfFilename = fileInfo.absoluteFilePath();
    }

    // Log startup information
    std::cout << betsee::AppMetadata::NAME << " "
              << betsee::AppMetadata::VERSION << std::endl;
    std::cout << betsee::AppMetadata::SYNOPSIS << std::endl;

    if (parser.isSet(verboseOption)) {
        std::cout << "Verbose logging enabled." << std::endl;
    }

    // Create and show the main window (port of gui/guimain.py)
    betsee::BetseeMainWindow mainWindow(simConfFilename);
    g_mainWindow = &mainWindow;

    // Log startup complete
    mainWindow.logViewer()->logInfo(
        QObject::tr("BETSEE %1 started successfully.")
            .arg(betsee::AppMetadata::VERSION));

    if (!simConfFilename.isEmpty()) {
        mainWindow.logViewer()->logInfo(
            QObject::tr("Opening initial configuration: %1")
                .arg(simConfFilename));
    }

    // Show the main window
    mainWindow.show();

    // Run the application event loop
    int result = app.exec();

    // Clean up
    g_mainWindow = nullptr;

    return result;
}

#else // !HAS_QT5

// ---------------------------------------------------------------------------
// Non-Qt fallback: print info and exit
// ---------------------------------------------------------------------------

#include <iostream>
#include <cstring>

int main(int argc, char* argv[]) {
    std::cout << betsee::AppMetadata::NAME << " "
              << betsee::AppMetadata::VERSION << std::endl;
    std::cout << betsee::AppMetadata::SYNOPSIS << std::endl;
    std::cout << std::endl;
    std::cout << betsee::AppMetadata::DESCRIPTION << std::endl;
    std::cout << std::endl;

    // Even without Qt, we can still test the YAML config parser
    // and simulation parameter management
    if (argc > 1) {
        if (std::strcmp(argv[1], "--help") == 0 || std::strcmp(argv[1], "-h") == 0) {
            std::cout << "Usage: betsee [options] [config_file]" << std::endl;
            std::cout << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  -h, --help              Show this help message" << std::endl;
            std::cout << "  -v, --version           Show version information" << std::endl;
            std::cout << "  -f, --sim-conf-file     Simulation configuration file" << std::endl;
            std::cout << "  --test-yaml FILE        Test YAML configuration parsing" << std::endl;
            std::cout << std::endl;
            std::cout << "This build was compiled without Qt5 support." << std::endl;
            std::cout << "Install Qt5 development libraries and rebuild with" << std::endl;
            std::cout << "-DHAS_QT5 to enable the full GUI." << std::endl;
            return 0;
        }

        if (std::strcmp(argv[1], "--version") == 0 || std::strcmp(argv[1], "-v") == 0) {
            std::cout << "Version: " << betsee::AppMetadata::VERSION << std::endl;
            std::cout << "Authors: " << betsee::AppMetadata::AUTHORS << std::endl;
            std::cout << "License: " << betsee::AppMetadata::LICENSE << std::endl;
            std::cout << "Homepage: " << betsee::AppMetadata::URL_HOMEPAGE << std::endl;
            return 0;
        }

        if (std::strcmp(argv[1], "--test-yaml") == 0 && argc > 2) {
            std::cout << "Testing YAML configuration parser..." << std::endl;
            std::cout << "Loading: " << argv[2] << std::endl;

            betsee::SimParameters params;
            if (params.load(argv[2])) {
                std::cout << "Successfully loaded configuration!" << std::endl;
                std::cout << std::endl;
                std::cout << "Parsed parameters:" << std::endl;
                std::cout << "  Init total time:     " << params.init_time_total << " s" << std::endl;
                std::cout << "  Init time step:      " << params.init_time_step << " s" << std::endl;
                std::cout << "  Init sampling rate:  " << params.init_time_sampling << " s" << std::endl;
                std::cout << "  Sim total time:      " << params.sim_time_total << " s" << std::endl;
                std::cout << "  Sim time step:       " << params.sim_time_step << " s" << std::endl;
                std::cout << "  Sim sampling rate:   " << params.sim_time_sampling << " s" << std::endl;
                std::cout << "  Cell radius:         " << params.cell_radius << " m" << std::endl;
                std::cout << "  Lattice disorder:    " << params.cell_lattice_disorder << std::endl;
                std::cout << "  Lattice type:        "
                          << (params.cell_lattice_type == betsee::CellLatticeType::HEX ? "hex" : "square")
                          << std::endl;
                std::cout << "  Grid size:           " << params.grid_size << std::endl;
                std::cout << "  ECM enabled:         " << (params.is_ecm ? "true" : "false") << std::endl;
                std::cout << "  World length:        " << params.world_len << " m" << std::endl;

                const char* ionProfileNames[] = {"basic", "basic_Ca", "mammal", "amphibian", "custom"};
                std::cout << "  Ion profile:         " << ionProfileNames[static_cast<int>(params.ion_profile)] << std::endl;

                // Test save
                std::string outFile = std::string(argv[2]) + ".test_out.yaml";
                if (params.save(outFile)) {
                    std::cout << std::endl;
                    std::cout << "Successfully saved to: " << outFile << std::endl;
                } else {
                    std::cerr << "Failed to save to: " << outFile << std::endl;
                }
            } else {
                std::cerr << "Failed to load configuration file: " << argv[2] << std::endl;
                return 1;
            }
            return 0;
        }

        // If a positional argument is given, try to parse it as a config file
        std::cout << "Attempting to parse: " << argv[1] << std::endl;
        betsee::SimParameters params;
        if (params.load(argv[1])) {
            std::cout << "Configuration parsed successfully." << std::endl;
            std::cout << "  Config file: " << params.conf_filename << std::endl;
            std::cout << "  Config dir:  " << params.conf_dirname << std::endl;
        } else {
            std::cerr << "Failed to parse configuration file." << std::endl;
            return 1;
        }
    } else {
        std::cout << "Build was compiled without Qt5 support." << std::endl;
        std::cout << "Run with --help for available options." << std::endl;
        std::cout << "Install Qt5 and rebuild with cmake -DHAS_QT5=ON for the full GUI." << std::endl;
    }

    return 0;
}

#endif // HAS_QT5
