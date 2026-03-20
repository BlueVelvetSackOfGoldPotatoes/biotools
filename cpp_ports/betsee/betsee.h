// ---------------------------------------------------------------------------
// betsee.h - BETSEE C++17/Qt5 Port
// BioElectric Tissue Simulation Engine Environment
// Original Python/PySide2 by Alexis Pietak & Cecil Curry.
// C++ port: complete reimplementation of the BETSEE GUI frontend for BETSE.
// ---------------------------------------------------------------------------
#ifndef BETSEE_H
#define BETSEE_H

// ---------------------------------------------------------------------------
// Qt detection and preprocessor guards
// ---------------------------------------------------------------------------
#ifdef HAS_QT5
#include <QApplication>
#include <QMainWindow>
#include <QWidget>
#include <QAction>
#include <QMenu>
#include <QMenuBar>
#include <QToolBar>
#include <QStatusBar>
#include <QDockWidget>
#include <QSplitter>
#include <QStackedWidget>
#include <QTabWidget>
#include <QTreeWidget>
#include <QTreeWidgetItem>
#include <QTextEdit>
#include <QPlainTextEdit>
#include <QLineEdit>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QCheckBox>
#include <QComboBox>
#include <QRadioButton>
#include <QGroupBox>
#include <QPushButton>
#include <QToolButton>
#include <QLabel>
#include <QProgressBar>
#include <QFrame>
#include <QScrollArea>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QFormLayout>
#include <QFileDialog>
#include <QMessageBox>
#include <QInputDialog>
#include <QDialog>
#include <QDialogButtonBox>
#include <QUndoStack>
#include <QUndoCommand>
#include <QSettings>
#include <QCloseEvent>
#include <QProcess>
#include <QThread>
#include <QTimer>
#include <QMutex>
#include <QSplashScreen>
#include <QPixmap>
#include <QIcon>
#include <QKeySequence>
#include <QClipboard>
#include <QMimeData>
#include <QSortFilterProxyModel>
#include <QHeaderView>
#include <QFileInfo>
#include <QDir>
#include <QStandardPaths>
#include <QDateTime>
#include <QRegularExpression>
#include <QFont>
#include <QColor>
#include <QBrush>
#include <QTextCursor>
#include <QTextCharFormat>
#include <QDesktopServices>
#include <QUrl>
#include <QDebug>
#else
// Stub types when Qt is not available, allowing compilation for analysis
#include <string>
#include <vector>
#include <map>
#include <functional>
#include <memory>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cassert>
#include <cstdint>
namespace QtStub {
    class QObject {
    public:
        QObject(QObject* parent = nullptr) { (void)parent; }
        virtual ~QObject() = default;
        void setObjectName(const std::string&) {}
        std::string objectName() const { return ""; }
    };
    class QWidget : public QObject {
    public:
        QWidget(QWidget* parent = nullptr) : QObject(parent) {}
        void show() {}
        void hide() {}
        void setEnabled(bool) {}
        void setVisible(bool) {}
        void setToolTip(const std::string&) {}
        void setMinimumSize(int, int) {}
        void resize(int, int) {}
        void setWindowTitle(const std::string&) {}
        void setFocusPolicy(int) {}
        void close() {}
        bool isEnabled() const { return true; }
    };
    class QMainWindow : public QWidget {
    public:
        QMainWindow(QWidget* parent = nullptr) : QWidget(parent) {}
        void setCentralWidget(QWidget*) {}
        void setWindowModified(bool) {}
    };
    class QAction : public QObject {
    public:
        QAction(QObject* parent = nullptr) : QObject(parent) {}
        QAction(const std::string&, QObject* parent = nullptr) : QObject(parent) {}
        void setEnabled(bool) {}
        void setChecked(bool) {}
        void setCheckable(bool) {}
        void setText(const std::string&) {}
        bool isEnabled() const { return true; }
        bool isSeparator() const { return false; }
    };
    class QString : public std::string {
    public:
        using std::string::string;
        QString() = default;
        QString(const std::string& s) : std::string(s) {}
        QString(const char* s) : std::string(s) {}
        bool isEmpty() const { return empty(); }
        std::string toStdString() const { return *this; }
        static QString fromStdString(const std::string& s) { return QString(s); }
        static QString number(int n) { return QString(std::to_string(n)); }
        static QString number(double n) { return QString(std::to_string(n)); }
    };
    using QStringList = std::vector<QString>;
    class QLayout {};
    class QVBoxLayout : public QLayout {};
    class QHBoxLayout : public QLayout {};
    class QGridLayout : public QLayout {};
    class QFormLayout : public QLayout {};
    class QLabel : public QWidget {};
    class QLineEdit : public QWidget {};
    class QTextEdit : public QWidget {};
    class QPlainTextEdit : public QWidget {};
    class QSpinBox : public QWidget {};
    class QDoubleSpinBox : public QWidget {};
    class QCheckBox : public QWidget {};
    class QComboBox : public QWidget {};
    class QRadioButton : public QWidget {};
    class QGroupBox : public QWidget {};
    class QPushButton : public QWidget {};
    class QToolButton : public QWidget {};
    class QProgressBar : public QWidget {};
    class QFrame : public QWidget {};
    class QScrollArea : public QWidget {};
    class QSplitter : public QWidget {};
    class QStackedWidget : public QWidget {};
    class QTabWidget : public QWidget {};
    class QTreeWidget : public QWidget {};
    class QTreeWidgetItem {};
    class QMenu : public QWidget {};
    class QMenuBar : public QWidget {};
    class QToolBar : public QWidget {};
    class QStatusBar : public QWidget {};
    class QDockWidget : public QWidget {};
    class QDialog : public QWidget {};
    class QUndoStack : public QObject {};
    class QUndoCommand {};
    class QProcess : public QObject {};
    class QThread : public QObject {};
    class QTimer : public QObject {};
    class QSettings {};
    class QCloseEvent {};
    class QSplashScreen : public QWidget {};
    class QFileDialog {};
    class QMessageBox {
    public:
        enum StandardButton { Ok = 1, Cancel = 2, Yes = 4, No = 8, Save = 16, Discard = 32 };
        enum Icon { NoIcon = 0, Information = 1, Warning = 2, Critical = 3, Question = 4 };
    };
    class QIcon {};
    class QPixmap {};
    class QColor {};
    class QFont {};
    class QKeySequence {};
    class QUrl { public: QUrl(const std::string&) {} };
    namespace Qt {
        constexpr int NoFocus = 0;
        constexpr int LeftDockWidgetArea = 1;
        constexpr int RightDockWidgetArea = 2;
        constexpr int BottomDockWidgetArea = 8;
        constexpr int AlignRight = 0x0002;
        constexpr int AlignLeft = 0x0001;
        constexpr int AlignCenter = 0x0004;
        constexpr int Horizontal = 0x1;
        constexpr int Vertical = 0x2;
        constexpr int UserRole = 0x0100;
    }
}
using namespace QtStub;
#endif // HAS_QT5

#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <memory>
#include <functional>
#include <deque>
#include <set>
#include <optional>
#include <variant>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <regex>
#include <cassert>
#include <atomic>
#include <mutex>

namespace betsee {

// ---------------------------------------------------------------------------
// Application metadata (port of betsee/guimetadata.py)
// ---------------------------------------------------------------------------
struct AppMetadata {
    static constexpr const char* NAME = "BETSEE";
    static constexpr const char* VERSION = "1.1.2.0";
    static constexpr const char* SYNOPSIS =
        "BETSEE, the BioElectric Tissue Simulation Engine Environment.";
    static constexpr const char* DESCRIPTION =
        "The BioElectric Tissue Simulation Engine Environment (BETSEE) is the "
        "official Qt 5-based graphical user interface (GUI) for BETSE, a "
        "finite volume simulator for 2D computational multiphysics problems in "
        "the life sciences -- including electrodiffusion, electro-osmosis, "
        "galvanotaxis, voltage-gated ion channels, gene regulatory networks, "
        "and biochemical reaction networks.";
    static constexpr const char* AUTHORS = "Alexis Pietak, Cecil Curry, et al.";
    static constexpr const char* AUTHOR_EMAIL = "leycec@gmail.com";
    static constexpr const char* LICENSE = "2-clause BSD";
    static constexpr const char* URL_HOMEPAGE = "https://gitlab.com/betse/betsee";
    static constexpr const char* ORG_NAME = "Paul Allen Discovery Center";
    static constexpr const char* ORG_DOMAIN = "alleninstitute.org";
};

// ---------------------------------------------------------------------------
// Forward declarations
// ---------------------------------------------------------------------------
class BetseeMainWindow;
class SimConf;
class SimConfUndoStack;
class SimConfTree;
class SimConfStack;
class SimmerTabWidget;
class Simmer;
class SimmerProactor;
class SimmerPhase;
class LogViewer;

// ---------------------------------------------------------------------------
// Enumerations (port of guisimrunenum.py)
// ---------------------------------------------------------------------------

enum class SimmerState {
    UNQUEUED, QUEUED, MODELLING, EXPORTING, PAUSED, STOPPING, FINISHED
};

enum class SimmerModelState { PREPARING, MODELLING, FINISHING };
enum class SimPhaseKind { SEED, INIT, SIM };
enum class SimmerPhaseSubkind { MODELLING, EXPORTING };
enum class SimExportType { CSV, PLOT_CELL, PLOT_CELLS, ANIM_CELLS };
enum class IonProfileType { BASIC, BASIC_CA, MAMMAL, AMPHIBIAN, CUSTOM };
enum class CellLatticeType { HEX, SQUARE };

// ---------------------------------------------------------------------------
// State sets and maps (port of guisimrunstate.py)
// ---------------------------------------------------------------------------

inline const std::set<SimmerState> SIMMER_STATES_IDLE = {
    SimmerState::UNQUEUED, SimmerState::QUEUED,
};
inline const std::set<SimmerState> SIMMER_STATES_RUNNING = {
    SimmerState::MODELLING, SimmerState::EXPORTING,
};
inline const std::set<SimmerState> SIMMER_STATES_WORKING = {
    SimmerState::MODELLING, SimmerState::EXPORTING, SimmerState::PAUSED,
};
inline const std::set<SimmerState> SIMMER_STATES_HALTING = {
    SimmerState::PAUSED, SimmerState::STOPPING, SimmerState::FINISHED,
};
inline const std::set<SimmerState> SIMMER_STATES_UNWORKABLE = {
    SimmerState::UNQUEUED, SimmerState::STOPPING,
};
inline const std::map<SimPhaseKind, std::string> SIM_PHASE_KIND_TO_NAME = {
    {SimPhaseKind::SEED, "seed"}, {SimPhaseKind::INIT, "initialization"},
    {SimPhaseKind::SIM,  "simulation"},
};
inline const std::map<SimmerState, std::string> SIMMER_STATE_TO_PHASE_STATUS = {
    {SimmerState::UNQUEUED,  "Unqueued"},  {SimmerState::QUEUED,    "Queued"},
    {SimmerState::MODELLING, "Modelling"}, {SimmerState::EXPORTING, "Exporting"},
    {SimmerState::PAUSED,    "Paused"},    {SimmerState::STOPPING,  "Finishing"},
    {SimmerState::FINISHED,  "Finished"},
};
inline const std::map<SimmerState, std::string> SIMMER_STATE_TO_PROACTOR_STATUS = {
    {SimmerState::UNQUEUED,  "Waiting for phase(s) to be queued..."},
    {SimmerState::QUEUED,    "Waiting for queued phase(s) to be started..."},
    {SimmerState::MODELLING, "Modelling <b>{phase_type}</b> phase..."},
    {SimmerState::EXPORTING, "Exporting <b>{phase_type}</b> phase..."},
    {SimmerState::PAUSED,    "Paused {status_prior}."},
    {SimmerState::STOPPING,  "Finishing {status_prior}..."},
    {SimmerState::FINISHED,  "Finished {status_prior}."},
};

// ---------------------------------------------------------------------------
// YAML Configuration Manager
// ---------------------------------------------------------------------------

struct YamlValue {
    std::string str_val;
    double num_val = 0.0;
    bool bool_val = false;
    enum Type { STRING, NUMBER, BOOLEAN } type = STRING;
    YamlValue() = default;
    explicit YamlValue(const std::string& s) : str_val(s), type(STRING) {}
    explicit YamlValue(double d) : num_val(d), type(NUMBER) {}
    explicit YamlValue(bool b) : bool_val(b), type(BOOLEAN) {}
};

class YamlConfig {
public:
    std::map<std::string, YamlValue> data_;
    std::string filename_;
    bool loaded_ = false;

    bool load(const std::string& filename) {
        filename_ = filename;
        std::ifstream ifs(filename);
        if (!ifs.is_open()) return false;
        data_.clear();
        std::string line, current_section;
        while (std::getline(ifs, line)) {
            auto trimmed = trim(line);
            if (trimmed.empty() || trimmed[0] == '#') continue;
            size_t indent = 0;
            for (char c : line) { if (c == ' ') indent++; else break; }
            auto colon_pos = trimmed.find(':');
            if (colon_pos != std::string::npos) {
                std::string key = trim(trimmed.substr(0, colon_pos));
                std::string val = (colon_pos + 1 < trimmed.size())
                    ? trim(trimmed.substr(colon_pos + 1)) : "";
                if (indent == 0) {
                    current_section = key;
                    if (!val.empty()) data_[key] = parseValue(val);
                } else {
                    std::string full_key = current_section + "." + key;
                    if (!val.empty()) data_[full_key] = parseValue(val);
                }
            }
        }
        loaded_ = true;
        return true;
    }

    bool save(const std::string& filename) const {
        std::ofstream ofs(filename);
        if (!ofs.is_open()) return false;
        std::string prev_section;
        for (auto& [key, val] : data_) {
            auto dot_pos = key.find('.');
            if (dot_pos != std::string::npos) {
                std::string section = key.substr(0, dot_pos);
                std::string subkey = key.substr(dot_pos + 1);
                if (section != prev_section) {
                    if (!prev_section.empty()) ofs << "\n";
                    ofs << section << ":\n";
                    prev_section = section;
                }
                ofs << "    " << subkey << ": " << valueToString(val) << "\n";
            } else {
                if (!prev_section.empty()) { ofs << "\n"; prev_section.clear(); }
                ofs << key << ": " << valueToString(val) << "\n";
            }
        }
        return true;
    }
    bool save() const { return save(filename_); }

    void setString(const std::string& key, const std::string& val) { data_[key] = YamlValue(val); }
    void setNumber(const std::string& key, double val) { data_[key] = YamlValue(val); }
    void setBool(const std::string& key, bool val) { data_[key] = YamlValue(val); }

    std::string getString(const std::string& key, const std::string& def = "") const {
        auto it = data_.find(key);
        return (it != data_.end()) ? it->second.str_val : def;
    }
    double getNumber(const std::string& key, double def = 0.0) const {
        auto it = data_.find(key);
        return (it != data_.end()) ? it->second.num_val : def;
    }
    bool getBool(const std::string& key, bool def = false) const {
        auto it = data_.find(key);
        return (it != data_.end()) ? it->second.bool_val : def;
    }
    bool isLoaded() const { return loaded_; }

private:
    static std::string trim(const std::string& s) {
        size_t start = s.find_first_not_of(" \t\r\n");
        size_t end = s.find_last_not_of(" \t\r\n");
        return (start == std::string::npos) ? "" : s.substr(start, end - start + 1);
    }
    static YamlValue parseValue(const std::string& s) {
        if (s == "true" || s == "True" || s == "TRUE") return YamlValue(true);
        if (s == "false" || s == "False" || s == "FALSE") return YamlValue(false);
        try { double d = std::stod(s); return YamlValue(d); } catch (...) {}
        if (s.size() >= 2 && ((s.front() == '\'' && s.back() == '\'') ||
                              (s.front() == '"' && s.back() == '"')))
            return YamlValue(s.substr(1, s.size() - 2));
        return YamlValue(s);
    }
    static std::string valueToString(const YamlValue& v) {
        switch (v.type) {
            case YamlValue::BOOLEAN: return v.bool_val ? "true" : "false";
            case YamlValue::NUMBER: { std::ostringstream o; o << v.num_val; return o.str(); }
            default: return v.str_val;
        }
    }
};

// ---------------------------------------------------------------------------
// Tissue profile and export item configurations
// ---------------------------------------------------------------------------

struct TissueProfile {
    std::string name = "default";
    std::string picker_image_filename;
    double Dm_Na = 1.0e-18, Dm_K = 1.0e-18, Dm_Cl = 1.0e-18;
    double Dm_Ca = 1.0e-18, Dm_M = 1.0e-18, Dm_P = 1.0e-18;
    bool is_enabled = true;
};

struct ExportItem {
    std::string name, type, pipeline_name;
    bool is_enabled = true;
    std::map<std::string, std::string> settings;
};

// ---------------------------------------------------------------------------
// Simulation parameters (port of betse.science.parameters.Parameters)
// ---------------------------------------------------------------------------

class SimParameters {
public:
    std::string conf_filename, conf_dirname;
    std::string seed_pickle_basename = "seed.betse.gz";
    std::string init_pickle_basename = "init.betse.gz";
    std::string init_pickle_dirname_relative = "init";
    std::string init_export_dirname_relative = "init_export";
    std::string sim_pickle_basename = "sim.betse.gz";
    std::string sim_pickle_dirname_relative = "sim";
    std::string sim_export_dirname_relative = "sim_export";

    double init_time_total = 10.0, init_time_step = 1.0e-3, init_time_sampling = 1.0;
    double sim_time_total = 100.0, sim_time_step = 1.0e-3, sim_time_sampling = 1.0;
    double cell_radius = 5.0e-6, cell_lattice_disorder = 0.4;
    CellLatticeType cell_lattice_type = CellLatticeType::HEX;
    int grid_size = 50;
    bool is_ecm = false;
    double world_len = 200.0e-6;
    IonProfileType ion_profile = IonProfileType::BASIC;

    // Custom ion concentrations (used when ion_profile == CUSTOM)
    double custom_Na_cell = 12.0, custom_Na_env = 145.0;
    double custom_K_cell = 140.0, custom_K_env = 5.0;
    double custom_Cl_cell = 4.0, custom_Cl_env = 110.0;
    double custom_Ca_cell = 0.0001, custom_Ca_env = 2.0;

    TissueProfile tissue_default;
    std::vector<TissueProfile> tissue_custom;

    struct AnimConfig {
        bool is_after_sim_show = true, is_after_sim_save = false;
        bool is_while_sim_show = false, is_while_sim_save = false;
        std::vector<ExportItem> anims_after_sim;
    } anim;
    struct PlotConfig {
        bool is_after_sim_show = true, is_after_sim_save = false;
        std::vector<ExportItem> plots_cell_after_sim, plots_cells_after_sim;
    } plot;
    struct CSVConfig { std::vector<ExportItem> csvs_after_sim; } csv;

    bool is_loaded = false;

    bool load(const std::string& filename) {
        yaml_.load(filename);
        conf_filename = filename;
        namespace fs = std::filesystem;
        conf_dirname = fs::path(filename).parent_path().string();
        syncFromYaml();
        is_loaded = true;
        return true;
    }
    bool save() { if (conf_filename.empty()) return false; syncToYaml(); return yaml_.save(conf_filename); }
    bool save(const std::string& fn) {
        conf_filename = fn;
        namespace fs = std::filesystem;
        conf_dirname = fs::path(fn).parent_path().string();
        syncToYaml();
        return yaml_.save(fn);
    }
    void unload() { is_loaded = false; conf_filename.clear(); conf_dirname.clear(); }

    SimParameters deepCopy() const {
        SimParameters c = *this;
        c.anim.is_after_sim_show = false; c.anim.is_while_sim_show = false;
        c.plot.is_after_sim_show = false;
        c.anim.is_after_sim_save = true; c.anim.is_while_sim_save = true;
        c.plot.is_after_sim_save = true;
        return c;
    }

private:
    YamlConfig yaml_;
    void syncFromYaml() {
        init_time_total = yaml_.getNumber("init time settings.total time", init_time_total);
        init_time_step = yaml_.getNumber("init time settings.time step", init_time_step);
        init_time_sampling = yaml_.getNumber("init time settings.sampling rate", init_time_sampling);
        sim_time_total = yaml_.getNumber("sim time settings.total time", sim_time_total);
        sim_time_step = yaml_.getNumber("sim time settings.time step", sim_time_step);
        sim_time_sampling = yaml_.getNumber("sim time settings.sampling rate", sim_time_sampling);
        cell_radius = yaml_.getNumber("cell cluster.cell radius", cell_radius);
        cell_lattice_disorder = yaml_.getNumber("cell cluster.lattice disorder", cell_lattice_disorder);
        grid_size = static_cast<int>(yaml_.getNumber("general.comp grid size", grid_size));
        is_ecm = yaml_.getBool("general.simulate extracellular spaces", is_ecm);
        world_len = yaml_.getNumber("general.world length", world_len);
        std::string lt = yaml_.getString("cell cluster.lattice type", "hex");
        cell_lattice_type = (lt == "square") ? CellLatticeType::SQUARE : CellLatticeType::HEX;
        std::string ip = yaml_.getString("general.ion profile", "basic");
        if (ip == "basic_Ca") ion_profile = IonProfileType::BASIC_CA;
        else if (ip == "mammal") ion_profile = IonProfileType::MAMMAL;
        else if (ip == "amphibian") ion_profile = IonProfileType::AMPHIBIAN;
        else if (ip == "custom") ion_profile = IonProfileType::CUSTOM;
        else ion_profile = IonProfileType::BASIC;
    }
    void syncToYaml() {
        yaml_.setNumber("init time settings.total time", init_time_total);
        yaml_.setNumber("init time settings.time step", init_time_step);
        yaml_.setNumber("init time settings.sampling rate", init_time_sampling);
        yaml_.setNumber("sim time settings.total time", sim_time_total);
        yaml_.setNumber("sim time settings.time step", sim_time_step);
        yaml_.setNumber("sim time settings.sampling rate", sim_time_sampling);
        yaml_.setNumber("cell cluster.cell radius", cell_radius);
        yaml_.setNumber("cell cluster.lattice disorder", cell_lattice_disorder);
        yaml_.setNumber("general.comp grid size", grid_size);
        yaml_.setBool("general.simulate extracellular spaces", is_ecm);
        yaml_.setNumber("general.world length", world_len);
        yaml_.setString("cell cluster.lattice type",
            cell_lattice_type == CellLatticeType::SQUARE ? "square" : "hex");
        const char* ips[] = {"basic","basic_Ca","mammal","amphibian","custom"};
        yaml_.setString("general.ion profile", ips[static_cast<int>(ion_profile)]);
    }
};

// ===========================================================================
// Qt-dependent classes (only compiled when HAS_QT5 is defined)
// ===========================================================================
#ifdef HAS_QT5

// ---------------------------------------------------------------------------
// Application-wide Signaler (port of gui/guimainsignaler.py)
// ---------------------------------------------------------------------------
class BetseeSignaler : public QObject {
    Q_OBJECT
public:
    explicit BetseeSignaler(QObject* parent = nullptr) : QObject(parent) {}
signals:
    void restoreSettingsSignal();
    void storeSettingsSignal();
};

// ---------------------------------------------------------------------------
// Log Viewer Widget (port of log/guilogconf.py)
// ---------------------------------------------------------------------------
class LogViewer : public QTextEdit {
    Q_OBJECT
public:
    explicit LogViewer(QWidget* parent = nullptr) : QTextEdit(parent) {
        setReadOnly(true);
        setObjectName("log_box");
        setMinimumHeight(100);
        QFont font("Monospace");
        font.setStyleHint(QFont::Monospace);
        font.setPointSize(9);
        setFont(font);
    }
    void appendLog(const QString& level, const QString& message) {
        QTextCursor cursor = textCursor();
        cursor.movePosition(QTextCursor::End);
        QTextCharFormat fmt;
        if (level == "ERROR" || level == "CRITICAL")
            fmt.setForeground(QBrush(QColor(200, 0, 0)));
        else if (level == "WARNING")
            fmt.setForeground(QBrush(QColor(180, 120, 0)));
        else if (level == "DEBUG")
            fmt.setForeground(QBrush(QColor(100, 100, 100)));
        else
            fmt.setForeground(QBrush(QColor(0, 0, 0)));
        QString ts = QDateTime::currentDateTime().toString("hh:mm:ss.zzz");
        cursor.insertText(QString("[%1] %2: %3\n").arg(ts, level, message), fmt);
        ensureCursorVisible();
    }
public slots:
    void logInfo(const QString& msg)    { appendLog("INFO", msg); }
    void logDebug(const QString& msg)   { appendLog("DEBUG", msg); }
    void logWarning(const QString& msg) { appendLog("WARNING", msg); }
    void logError(const QString& msg)   { appendLog("ERROR", msg); }
    void clearLog() { clear(); }
};

// ---------------------------------------------------------------------------
// Settings (port of util/io/guisettings.py)
// ---------------------------------------------------------------------------
class BetseeSettings : public QObject {
    Q_OBJECT
public:
    explicit BetseeSettings(QObject* parent = nullptr) : QObject(parent) {
        settings_ = std::make_unique<QSettings>(
            QSettings::IniFormat, QSettings::UserScope,
            AppMetadata::ORG_DOMAIN, AppMetadata::NAME);
    }
    QVariant value(const QString& key, const QVariant& def = QVariant()) const {
        return settings_->value(key, def);
    }
    void setValue(const QString& key, const QVariant& val) { settings_->setValue(key, val); }
public slots:
    void restoreSettings() {
        // Restore window geometry
        if (settings_->contains("mainwindow/geometry")) {
            auto* mw = qobject_cast<QMainWindow*>(qApp->activeWindow());
            if (mw) {
                mw->restoreGeometry(settings_->value("mainwindow/geometry").toByteArray());
                mw->restoreState(settings_->value("mainwindow/state").toByteArray());
            }
        }
        // Restore last opened directory
        lastDir_ = settings_->value("files/lastDir", QDir::homePath()).toString();
        // Restore recent files list
        recentFiles_ = settings_->value("files/recent").toStringList();
    }
    void storeSettings() {
        auto* mw = qobject_cast<QMainWindow*>(qApp->activeWindow());
        if (mw) {
            settings_->setValue("mainwindow/geometry", mw->saveGeometry());
            settings_->setValue("mainwindow/state", mw->saveState());
        }
        settings_->setValue("files/lastDir", lastDir_);
        settings_->setValue("files/recent", recentFiles_);
        settings_->sync();
    }
    QString lastDir() const { return lastDir_; }
    void setLastDir(const QString& d) { lastDir_ = d; }
    QStringList recentFiles() const { return recentFiles_; }
    void addRecentFile(const QString& f) {
        recentFiles_.removeAll(f);
        recentFiles_.prepend(f);
        while (recentFiles_.size() > 10) recentFiles_.removeLast();
    }
private:
    QString lastDir_;
    QStringList recentFiles_;
    std::unique_ptr<QSettings> settings_;
};

// ---------------------------------------------------------------------------
// Undo Command for simulation configuration edits
// ---------------------------------------------------------------------------
class SimConfUndoCommand : public QUndoCommand {
public:
    SimConfUndoCommand(const QString& text, const QString& key,
                       const QVariant& oldVal, const QVariant& newVal,
                       std::function<void(const QString&, const QVariant&)> applier,
                       QUndoCommand* parent = nullptr)
        : QUndoCommand(text, parent), key_(key), oldVal_(oldVal),
          newVal_(newVal), applier_(std::move(applier)) {}
    void undo() override { if (applier_) applier_(key_, oldVal_); }
    void redo() override { if (applier_) applier_(key_, newVal_); }
private:
    QString key_;
    QVariant oldVal_, newVal_;
    std::function<void(const QString&, const QVariant&)> applier_;
};

// ---------------------------------------------------------------------------
// SimConf Undo Stack (port of gui/simconf/guisimconfundo.py)
// ---------------------------------------------------------------------------
class SimConfUndoStack : public QUndoStack {
    Q_OBJECT
public:
    explicit SimConfUndoStack(QObject* parent = nullptr) : QUndoStack(parent) {}

    void init(QMainWindow* mainWindow) {
        undoAction_ = createUndoAction(this, tr("&Undo"));
        undoAction_->setShortcuts(QKeySequence::Undo);
        undoAction_->setObjectName("action_undo");

        redoAction_ = createRedoAction(this, tr("&Redo"));
        redoAction_->setShortcuts(QKeySequence::Redo);
        redoAction_->setObjectName("action_redo");

        // Insert into Edit menu
        if (auto* mb = mainWindow->menuBar()) {
            for (auto* action : mb->actions()) {
                if (auto* menu = action->menu()) {
                    if (menu->title().contains("Edit", Qt::CaseInsensitive)) {
                        auto acts = menu->actions();
                        if (!acts.isEmpty()) {
                            menu->insertAction(acts.first(), redoAction_);
                            menu->insertAction(redoAction_, undoAction_);
                        } else {
                            menu->addAction(undoAction_);
                            menu->addAction(redoAction_);
                        }
                        break;
                    }
                }
            }
        }
        // Insert into toolbar
        auto* toolbar = mainWindow->findChild<QToolBar*>("toolbar");
        if (toolbar) {
            QAction* firstSep = nullptr;
            for (auto* act : toolbar->actions()) {
                if (act->isSeparator()) { firstSep = act; break; }
            }
            if (firstSep) {
                toolbar->insertSeparator(firstSep);
                toolbar->insertAction(firstSep, undoAction_);
                toolbar->insertAction(firstSep, redoAction_);
            } else {
                toolbar->addSeparator();
                toolbar->addAction(undoAction_);
                toolbar->addAction(redoAction_);
            }
        }
    }

    void pushIfSafe(QUndoCommand* cmd) {
        if (open_) push(cmd); else delete cmd;
    }
    void setSimConfOpen(bool open) { open_ = open; }

private:
    QAction* undoAction_ = nullptr;
    QAction* redoAction_ = nullptr;
    bool open_ = false;
};

// ---------------------------------------------------------------------------
// Simulation Configuration Editor Widgets
// ---------------------------------------------------------------------------

/// Sim-conf aware double spin box
class SimConfDoubleSpinBox : public QDoubleSpinBox {
    Q_OBJECT
public:
    explicit SimConfDoubleSpinBox(QWidget* parent = nullptr)
        : QDoubleSpinBox(parent) {
        setAccelerated(true);
        setAlignment(Qt::AlignRight);
        setKeyboardTracking(false);
        setDecimals(6);
    }
    void initParam(SimParameters* p, double* ptr,
                   SimConfUndoStack* undo, const QString& name) {
        params_ = p; ptr_ = ptr; undo_ = undo; name_ = name;
        if (ptr_) setValue(*ptr_);
        connect(this, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
                this, &SimConfDoubleSpinBox::onChanged);
    }
    void syncFromConfig() {
        if (ptr_) { blockSignals(true); setValue(*ptr_); blockSignals(false); }
    }
private slots:
    void onChanged(double v) {
        if (!ptr_ || !params_) return;
        double old = *ptr_;
        if (old == v) return;
        *ptr_ = v;
        if (undo_) {
            undo_->pushIfSafe(new SimConfUndoCommand(
                tr("Edit %1").arg(name_), name_, QVariant(old), QVariant(v),
                [this](const QString&, const QVariant& val) {
                    if (ptr_) { *ptr_ = val.toDouble(); blockSignals(true); setValue(val.toDouble()); blockSignals(false); }
                }));
        }
        emit parameterChanged();
    }
signals:
    void parameterChanged();
private:
    SimParameters* params_ = nullptr;
    double* ptr_ = nullptr;
    SimConfUndoStack* undo_ = nullptr;
    QString name_;
};

/// Sim-conf aware integer spin box
class SimConfIntSpinBox : public QSpinBox {
    Q_OBJECT
public:
    explicit SimConfIntSpinBox(QWidget* parent = nullptr) : QSpinBox(parent) {
        setAccelerated(true); setAlignment(Qt::AlignRight); setKeyboardTracking(false);
    }
    void initParam(SimParameters* p, int* ptr,
                   SimConfUndoStack* undo, const QString& name) {
        params_ = p; ptr_ = ptr; undo_ = undo; name_ = name;
        if (ptr_) setValue(*ptr_);
        connect(this, QOverload<int>::of(&QSpinBox::valueChanged),
                this, &SimConfIntSpinBox::onChanged);
    }
    void syncFromConfig() {
        if (ptr_) { blockSignals(true); setValue(*ptr_); blockSignals(false); }
    }
private slots:
    void onChanged(int v) {
        if (!ptr_ || !params_) return;
        int old = *ptr_;
        if (old == v) return;
        *ptr_ = v;
        if (undo_) {
            undo_->pushIfSafe(new SimConfUndoCommand(
                tr("Edit %1").arg(name_), name_, QVariant(old), QVariant(v),
                [this](const QString&, const QVariant& val) {
                    if (ptr_) { *ptr_ = val.toInt(); blockSignals(true); setValue(val.toInt()); blockSignals(false); }
                }));
        }
        emit parameterChanged();
    }
signals:
    void parameterChanged();
private:
    SimParameters* params_ = nullptr;
    int* ptr_ = nullptr;
    SimConfUndoStack* undo_ = nullptr;
    QString name_;
};

/// Sim-conf aware checkbox
class SimConfCheckBox : public QCheckBox {
    Q_OBJECT
public:
    explicit SimConfCheckBox(const QString& text = "", QWidget* parent = nullptr)
        : QCheckBox(text, parent) {}
    void initParam(SimParameters* p, bool* ptr,
                   SimConfUndoStack* undo, const QString& name) {
        params_ = p; ptr_ = ptr; undo_ = undo; name_ = name;
        if (ptr_) setChecked(*ptr_);
        connect(this, &QCheckBox::toggled, this, &SimConfCheckBox::onToggled);
    }
    void syncFromConfig() {
        if (ptr_) { blockSignals(true); setChecked(*ptr_); blockSignals(false); }
    }
private slots:
    void onToggled(bool v) {
        if (!ptr_) return;
        bool old = *ptr_; *ptr_ = v;
        if (undo_) {
            undo_->pushIfSafe(new SimConfUndoCommand(
                tr("Toggle %1").arg(name_), name_, QVariant(old), QVariant(v),
                [this](const QString&, const QVariant& val) {
                    if (ptr_) { *ptr_ = val.toBool(); blockSignals(true); setChecked(val.toBool()); blockSignals(false); }
                }));
        }
        emit parameterChanged();
    }
signals:
    void parameterChanged();
private:
    SimParameters* params_ = nullptr;
    bool* ptr_ = nullptr;
    SimConfUndoStack* undo_ = nullptr;
    QString name_;
};

/// Sim-conf aware line edit
class SimConfLineEdit : public QLineEdit {
    Q_OBJECT
public:
    explicit SimConfLineEdit(QWidget* parent = nullptr) : QLineEdit(parent) {}
    void initParam(SimParameters* p, std::string* ptr,
                   SimConfUndoStack* undo, const QString& name,
                   QPushButton* browseBtn = nullptr) {
        params_ = p; ptr_ = ptr; undo_ = undo; name_ = name;
        if (ptr_) setText(QString::fromStdString(*ptr_));
        connect(this, &QLineEdit::editingFinished, this, &SimConfLineEdit::onFinished);
        if (browseBtn)
            connect(browseBtn, &QPushButton::clicked, this, &SimConfLineEdit::onBrowse);
    }
    void syncFromConfig() {
        if (ptr_) { blockSignals(true); setText(QString::fromStdString(*ptr_)); blockSignals(false); }
    }
private slots:
    void onFinished() {
        if (!ptr_) return;
        std::string nv = text().toStdString(), ov = *ptr_;
        if (ov == nv) return;
        *ptr_ = nv;
        if (undo_) {
            undo_->pushIfSafe(new SimConfUndoCommand(
                tr("Edit %1").arg(name_), name_,
                QVariant(QString::fromStdString(ov)),
                QVariant(QString::fromStdString(nv)),
                [this](const QString&, const QVariant& val) {
                    if (ptr_) { *ptr_ = val.toString().toStdString(); blockSignals(true); setText(val.toString()); blockSignals(false); }
                }));
        }
        emit parameterChanged();
    }
    void onBrowse() {
        QString dir = QFileDialog::getExistingDirectory(this, tr("Select Directory"),
            ptr_ ? QString::fromStdString(*ptr_) : QString());
        if (!dir.isEmpty()) { setText(dir); onFinished(); }
    }
signals:
    void parameterChanged();
private:
    SimParameters* params_ = nullptr;
    std::string* ptr_ = nullptr;
    SimConfUndoStack* undo_ = nullptr;
    QString name_;
};

/// File path editor with browse button
class SimConfFilePathEdit : public QWidget {
    Q_OBJECT
public:
    explicit SimConfFilePathEdit(QWidget* parent = nullptr) : QWidget(parent) {
        auto* lay = new QHBoxLayout(this);
        lay->setContentsMargins(0, 0, 0, 0);
        edit_ = new QLineEdit(this);
        btn_ = new QPushButton(tr("Browse..."), this);
        lay->addWidget(edit_, 1);
        lay->addWidget(btn_);
        connect(btn_, &QPushButton::clicked, this, &SimConfFilePathEdit::browse);
        connect(edit_, &QLineEdit::editingFinished, this, &SimConfFilePathEdit::finished);
    }
    void initParam(SimParameters*, std::string* ptr, const QString& name,
                   const QString& filter = "") {
        ptr_ = ptr; filter_ = filter;
        (void)name;
        if (ptr_) edit_->setText(QString::fromStdString(*ptr_));
    }
    void syncFromConfig() {
        if (ptr_) { edit_->blockSignals(true); edit_->setText(QString::fromStdString(*ptr_)); edit_->blockSignals(false); }
    }
private slots:
    void browse() {
        QString f = QFileDialog::getOpenFileName(this, tr("Select File"),
            ptr_ ? QString::fromStdString(*ptr_) : QString(), filter_);
        if (!f.isEmpty()) { edit_->setText(f); finished(); }
    }
    void finished() { if (ptr_) { *ptr_ = edit_->text().toStdString(); emit parameterChanged(); } }
signals:
    void parameterChanged();
private:
    QLineEdit* edit_;
    QPushButton* btn_;
    std::string* ptr_ = nullptr;
    QString filter_;
};

// ---------------------------------------------------------------------------
// SimConf Tree Widget (port of gui/simconf/tree/guisimconftree.py)
// ---------------------------------------------------------------------------
class SimConfTree : public QTreeWidget {
    Q_OBJECT
public:
    explicit SimConfTree(QWidget* parent = nullptr) : QTreeWidget(parent) {
        setObjectName("sim_conf_tree");
        setHeaderHidden(true);
        setColumnCount(1);
        setMinimumWidth(200);
        buildTree();
    }
    void init(SimParameters* p) {
        params_ = p; expandAll();
        setContextMenuPolicy(Qt::CustomContextMenu);
        connect(this, &QWidget::customContextMenuRequested, this, &SimConfTree::onContextMenu);
    }

    // Add a new custom tissue profile
    void addTissueProfile() {
        if (!params_) return;
        TissueProfile tp;
        tp.name = "Custom Profile " + std::to_string(params_->tissue_custom.size() + 1);
        params_->tissue_custom.push_back(tp);
        rebuildDynamic();
    }

    // Remove the currently selected custom tissue profile
    void removeCurrentTissueProfile(int index) {
        if (!params_ || index < 0 || index >= (int)params_->tissue_custom.size()) return;
        params_->tissue_custom.erase(params_->tissue_custom.begin() + index);
        rebuildDynamic();
    }

    // Add export item to a category
    void addExportItem(const QString& category) {
        if (!params_) return;
        ExportItem item;
        if (category == "anim") {
            item.name = "Animation " + std::to_string(params_->anim.anims_after_sim.size() + 1);
            params_->anim.anims_after_sim.push_back(item);
        } else if (category == "plot_cell") {
            item.name = "Plot " + std::to_string(params_->plot.plots_cell_after_sim.size() + 1);
            params_->plot.plots_cell_after_sim.push_back(item);
        } else if (category == "plot_cells") {
            item.name = "Plot " + std::to_string(params_->plot.plots_cells_after_sim.size() + 1);
            params_->plot.plots_cells_after_sim.push_back(item);
        } else if (category == "csv") {
            item.name = "CSV " + std::to_string(params_->csv.csvs_after_sim.size() + 1);
            params_->csv.csvs_after_sim.push_back(item);
        }
        rebuildDynamic();
    }
    void setSimConfOpen(bool open) {
        setEnabled(open);
        if (open && params_ && params_->is_loaded) rebuildDynamic();
    }
signals:
    void pageChangeRequested(const QString& pageName);
private slots:
    void onCurrent(QTreeWidgetItem* cur, QTreeWidgetItem*) {
        if (!cur) return;
        QString pg = cur->data(0, Qt::UserRole).toString();
        if (!pg.isEmpty()) emit pageChangeRequested(pg);
    }
    void onContextMenu(const QPoint& pos) {
        auto* item = itemAt(pos);
        if (!item || !params_) return;
        QString pg = item->data(0, Qt::UserRole).toString();
        QMenu menu;
        // Context menu for tissue node
        if (pg == "page_tissue_default" || pg.startsWith("page_tissue_custom_")) {
            menu.addAction(tr("Add Tissue Profile"), this, &SimConfTree::addTissueProfile);
            if (pg.startsWith("page_tissue_custom_")) {
                int idx = pg.mid(QString("page_tissue_custom_").length()).toInt();
                menu.addAction(tr("Remove This Profile"), this, [this, idx]() { removeCurrentTissueProfile(idx); });
            }
        }
        // Context menu for export nodes
        else if (pg == "page_anim" || pg == "page_anim_cells" || pg.startsWith("page_anim_cells_item_")) {
            menu.addAction(tr("Add Animation"), this, [this]() { addExportItem("anim"); });
        }
        else if (pg == "page_plot_cell" || pg.startsWith("page_plot_cell_item_")) {
            menu.addAction(tr("Add Single Cell Plot"), this, [this]() { addExportItem("plot_cell"); });
        }
        else if (pg == "page_plot_cells" || pg.startsWith("page_plot_cells_item_")) {
            menu.addAction(tr("Add Cell Cluster Plot"), this, [this]() { addExportItem("plot_cells"); });
        }
        else if (pg == "page_csv" || pg.startsWith("page_csv_item_")) {
            menu.addAction(tr("Add CSV Export"), this, [this]() { addExportItem("csv"); });
        }
        if (!menu.isEmpty()) menu.exec(viewport()->mapToGlobal(pos));
    }
private:
    SimParameters* params_ = nullptr;
    QTreeWidgetItem *tissueRoot_ = nullptr, *animRoot_ = nullptr;
    QTreeWidgetItem *plotCellRoot_ = nullptr, *plotCellsRoot_ = nullptr, *csvRoot_ = nullptr;

    void buildTree() {
        auto* paths = new QTreeWidgetItem(this, QStringList() << tr("Paths"));
        paths->setData(0, Qt::UserRole, "page_paths");
        auto* time = new QTreeWidgetItem(this, QStringList() << tr("Time"));
        time->setData(0, Qt::UserRole, "page_time");
        auto* space = new QTreeWidgetItem(this, QStringList() << tr("Space"));
        space->setData(0, Qt::UserRole, "page_space");
        auto* ions = new QTreeWidgetItem(space, QStringList() << tr("Ions"));
        ions->setData(0, Qt::UserRole, "page_ions");
        auto* tissue = new QTreeWidgetItem(space, QStringList() << tr("Tissue"));
        tissue->setData(0, Qt::UserRole, "page_tissue_default");
        tissueRoot_ = tissue;
        auto* exp = new QTreeWidgetItem(this, QStringList() << tr("Export"));
        exp->setData(0, Qt::UserRole, "page_export");
        auto* anim = new QTreeWidgetItem(exp, QStringList() << tr("Animations"));
        anim->setData(0, Qt::UserRole, "page_anim");
        auto* animCells = new QTreeWidgetItem(anim, QStringList() << tr("Cell Cluster"));
        animCells->setData(0, Qt::UserRole, "page_anim_cells");
        animRoot_ = animCells;
        auto* plotN = new QTreeWidgetItem(exp, QStringList() << tr("Plots"));
        plotN->setData(0, Qt::UserRole, "page_plot");
        auto* plotCell = new QTreeWidgetItem(plotN, QStringList() << tr("Single Cell"));
        plotCell->setData(0, Qt::UserRole, "page_plot_cell");
        plotCellRoot_ = plotCell;
        auto* plotCells = new QTreeWidgetItem(plotN, QStringList() << tr("Cell Cluster"));
        plotCells->setData(0, Qt::UserRole, "page_plot_cells");
        plotCellsRoot_ = plotCells;
        auto* csvN = new QTreeWidgetItem(exp, QStringList() << tr("CSV"));
        csvN->setData(0, Qt::UserRole, "page_csv");
        csvRoot_ = csvN;
        connect(this, &QTreeWidget::currentItemChanged, this, &SimConfTree::onCurrent);
        if (topLevelItemCount() > 0) setCurrentItem(topLevelItem(0));
    }

    void clearChildren(QTreeWidgetItem* item) {
        while (item->childCount() > 0) delete item->takeChild(0);
    }

    void rebuildDynamic() {
        if (!params_) return;
        clearChildren(tissueRoot_);
        for (size_t i = 0; i < params_->tissue_custom.size(); ++i) {
            auto* it = new QTreeWidgetItem(tissueRoot_);
            it->setText(0, QString::fromStdString(params_->tissue_custom[i].name));
            it->setData(0, Qt::UserRole, QString("page_tissue_custom_%1").arg(i));
        }
        clearChildren(animRoot_);
        for (size_t i = 0; i < params_->anim.anims_after_sim.size(); ++i) {
            auto* it = new QTreeWidgetItem(animRoot_);
            it->setText(0, QString::fromStdString(params_->anim.anims_after_sim[i].name));
            it->setData(0, Qt::UserRole, QString("page_anim_cells_item_%1").arg(i));
        }
        clearChildren(plotCellRoot_);
        for (size_t i = 0; i < params_->plot.plots_cell_after_sim.size(); ++i) {
            auto* it = new QTreeWidgetItem(plotCellRoot_);
            it->setText(0, QString::fromStdString(params_->plot.plots_cell_after_sim[i].name));
            it->setData(0, Qt::UserRole, QString("page_plot_cell_item_%1").arg(i));
        }
        clearChildren(plotCellsRoot_);
        for (size_t i = 0; i < params_->plot.plots_cells_after_sim.size(); ++i) {
            auto* it = new QTreeWidgetItem(plotCellsRoot_);
            it->setText(0, QString::fromStdString(params_->plot.plots_cells_after_sim[i].name));
            it->setData(0, Qt::UserRole, QString("page_plot_cells_item_%1").arg(i));
        }
        clearChildren(csvRoot_);
        for (size_t i = 0; i < params_->csv.csvs_after_sim.size(); ++i) {
            auto* it = new QTreeWidgetItem(csvRoot_);
            it->setText(0, QString::fromStdString(params_->csv.csvs_after_sim[i].name));
            it->setData(0, Qt::UserRole, QString("page_csv_item_%1").arg(i));
        }
        expandAll();
    }
};

// ---------------------------------------------------------------------------
// Stack pages for simulation configuration
// ---------------------------------------------------------------------------

class SimConfPagePath : public QScrollArea {
    Q_OBJECT
public:
    explicit SimConfPagePath(QWidget* parent = nullptr) : QScrollArea(parent) {
        setObjectName("page_paths"); setWidgetResizable(true);
        auto* c = new QWidget(this); auto* l = new QFormLayout(c);
        auto* sg = new QGroupBox(tr("Seed Phase Paths"), c);
        auto* sl = new QFormLayout(sg);
        seedFile_ = new SimConfLineEdit(sg); sl->addRow(tr("Seed Pickle File:"), seedFile_);
        l->addRow(sg);
        auto* ig = new QGroupBox(tr("Initialization Phase Paths"), c);
        auto* il = new QFormLayout(ig);
        initFile_ = new SimConfLineEdit(ig);
        initDir_ = new SimConfLineEdit(ig); initDirBtn_ = new QPushButton(tr("Browse..."), ig);
        initExpDir_ = new SimConfLineEdit(ig); initExpDirBtn_ = new QPushButton(tr("Browse..."), ig);
        auto* r1 = new QHBoxLayout(); r1->addWidget(initDir_, 1); r1->addWidget(initDirBtn_);
        auto* r2 = new QHBoxLayout(); r2->addWidget(initExpDir_, 1); r2->addWidget(initExpDirBtn_);
        il->addRow(tr("Init Pickle File:"), initFile_);
        il->addRow(tr("Init Pickle Dir:"), r1);
        il->addRow(tr("Init Export Dir:"), r2);
        l->addRow(ig);
        auto* smg = new QGroupBox(tr("Simulation Phase Paths"), c);
        auto* sml = new QFormLayout(smg);
        simFile_ = new SimConfLineEdit(smg);
        simDir_ = new SimConfLineEdit(smg); simDirBtn_ = new QPushButton(tr("Browse..."), smg);
        simExpDir_ = new SimConfLineEdit(smg); simExpDirBtn_ = new QPushButton(tr("Browse..."), smg);
        auto* r3 = new QHBoxLayout(); r3->addWidget(simDir_, 1); r3->addWidget(simDirBtn_);
        auto* r4 = new QHBoxLayout(); r4->addWidget(simExpDir_, 1); r4->addWidget(simExpDirBtn_);
        sml->addRow(tr("Sim Pickle File:"), simFile_);
        sml->addRow(tr("Sim Pickle Dir:"), r3);
        sml->addRow(tr("Sim Export Dir:"), r4);
        l->addRow(smg);
        setWidget(c);
    }
    void init(SimParameters* p, SimConfUndoStack* u) {
        seedFile_->initParam(p, &p->seed_pickle_basename, u, tr("Seed File"));
        initFile_->initParam(p, &p->init_pickle_basename, u, tr("Init File"));
        initDir_->initParam(p, &p->init_pickle_dirname_relative, u, tr("Init Dir"), initDirBtn_);
        initExpDir_->initParam(p, &p->init_export_dirname_relative, u, tr("Init Exp Dir"), initExpDirBtn_);
        simFile_->initParam(p, &p->sim_pickle_basename, u, tr("Sim File"));
        simDir_->initParam(p, &p->sim_pickle_dirname_relative, u, tr("Sim Dir"), simDirBtn_);
        simExpDir_->initParam(p, &p->sim_export_dirname_relative, u, tr("Sim Exp Dir"), simExpDirBtn_);
    }
    void syncFromConfig() {
        for (auto* w : {seedFile_,initFile_,initDir_,initExpDir_,simFile_,simDir_,simExpDir_}) w->syncFromConfig();
    }
private:
    SimConfLineEdit *seedFile_, *initFile_, *initDir_, *initExpDir_, *simFile_, *simDir_, *simExpDir_;
    QPushButton *initDirBtn_, *initExpDirBtn_, *simDirBtn_, *simExpDirBtn_;
};

class SimConfPageTime : public QScrollArea {
    Q_OBJECT
public:
    explicit SimConfPageTime(QWidget* parent = nullptr) : QScrollArea(parent) {
        setObjectName("page_time"); setWidgetResizable(true);
        auto* c = new QWidget(this); auto* l = new QVBoxLayout(c);
        auto* ig = new QGroupBox(tr("Initialization Time Settings"), c);
        auto* il = new QFormLayout(ig);
        itt_ = new SimConfDoubleSpinBox(ig); itt_->setRange(0,1e12); itt_->setSuffix(" s");
        its_ = new SimConfDoubleSpinBox(ig); its_->setRange(1e-12,1e6); its_->setSuffix(" s");
        itr_ = new SimConfDoubleSpinBox(ig); itr_->setRange(1e-6,1e6); itr_->setSuffix(" s");
        il->addRow(tr("Total Time:"), itt_); il->addRow(tr("Time Step:"), its_); il->addRow(tr("Sampling Rate:"), itr_);
        l->addWidget(ig);
        auto* sg = new QGroupBox(tr("Simulation Time Settings"), c);
        auto* sl = new QFormLayout(sg);
        stt_ = new SimConfDoubleSpinBox(sg); stt_->setRange(0,1e12); stt_->setSuffix(" s");
        sts_ = new SimConfDoubleSpinBox(sg); sts_->setRange(1e-12,1e6); sts_->setSuffix(" s");
        str_ = new SimConfDoubleSpinBox(sg); str_->setRange(1e-6,1e6); str_->setSuffix(" s");
        sl->addRow(tr("Total Time:"), stt_); sl->addRow(tr("Time Step:"), sts_); sl->addRow(tr("Sampling Rate:"), str_);
        l->addWidget(sg); l->addStretch();
        setWidget(c);
    }
    void init(SimParameters* p, SimConfUndoStack* u) {
        itt_->initParam(p, &p->init_time_total, u, tr("Init Total Time"));
        its_->initParam(p, &p->init_time_step, u, tr("Init Time Step"));
        itr_->initParam(p, &p->init_time_sampling, u, tr("Init Sampling Rate"));
        stt_->initParam(p, &p->sim_time_total, u, tr("Sim Total Time"));
        sts_->initParam(p, &p->sim_time_step, u, tr("Sim Time Step"));
        str_->initParam(p, &p->sim_time_sampling, u, tr("Sim Sampling Rate"));
    }
    void syncFromConfig() { for (auto* w : {itt_,its_,itr_,stt_,sts_,str_}) w->syncFromConfig(); }
private:
    SimConfDoubleSpinBox *itt_, *its_, *itr_, *stt_, *sts_, *str_;
};

class SimConfPageSpace : public QScrollArea {
    Q_OBJECT
public:
    explicit SimConfPageSpace(QWidget* parent = nullptr) : QScrollArea(parent) {
        setObjectName("page_space"); setWidgetResizable(true);
        auto* c = new QWidget(this); auto* l = new QVBoxLayout(c);
        auto* ig = new QGroupBox(tr("Intracellular Settings"), c);
        auto* il = new QFormLayout(ig);
        cr_ = new SimConfDoubleSpinBox(ig); cr_->setRange(1e-9,1e-3); cr_->setSuffix(" m");
        ld_ = new SimConfDoubleSpinBox(ig); ld_->setRange(0,1);
        hexR_ = new QRadioButton(tr("Hexagonal"), ig);
        sqR_ = new QRadioButton(tr("Square"), ig);
        auto* lr = new QHBoxLayout(); lr->addWidget(hexR_); lr->addWidget(sqR_);
        il->addRow(tr("Cell Radius:"), cr_); il->addRow(tr("Lattice Disorder:"), ld_);
        il->addRow(tr("Lattice Type:"), lr);
        l->addWidget(ig);
        auto* eg = new QGroupBox(tr("Extracellular Settings"), c);
        auto* el = new QFormLayout(eg);
        gs_ = new SimConfIntSpinBox(eg); gs_->setRange(10,10000);
        ecm_ = new SimConfCheckBox(tr("Simulate Extracellular Spaces"), eg);
        wl_ = new SimConfDoubleSpinBox(eg); wl_->setRange(1e-9,1); wl_->setSuffix(" m");
        el->addRow(tr("Grid Size:"), gs_); el->addRow("", ecm_); el->addRow(tr("World Length:"), wl_);
        l->addWidget(eg); l->addStretch();
        setWidget(c);
        connect(hexR_, &QRadioButton::toggled, this, [this](bool v) {
            if (v && p_) { p_->cell_lattice_type = CellLatticeType::HEX; emit parameterChanged(); }
        });
        connect(sqR_, &QRadioButton::toggled, this, [this](bool v) {
            if (v && p_) { p_->cell_lattice_type = CellLatticeType::SQUARE; emit parameterChanged(); }
        });
    }
    void init(SimParameters* p, SimConfUndoStack* u) {
        p_ = p;
        cr_->initParam(p, &p->cell_radius, u, tr("Cell Radius"));
        ld_->initParam(p, &p->cell_lattice_disorder, u, tr("Lattice Disorder"));
        gs_->initParam(p, &p->grid_size, u, tr("Grid Size"));
        ecm_->initParam(p, &p->is_ecm, u, tr("Simulate ECM"));
        wl_->initParam(p, &p->world_len, u, tr("World Length"));
    }
    void syncFromConfig() {
        cr_->syncFromConfig(); ld_->syncFromConfig(); gs_->syncFromConfig();
        ecm_->syncFromConfig(); wl_->syncFromConfig();
        if (p_) { blockSignals(true);
            hexR_->setChecked(p_->cell_lattice_type == CellLatticeType::HEX);
            sqR_->setChecked(p_->cell_lattice_type == CellLatticeType::SQUARE);
            blockSignals(false);
        }
    }
signals:
    void parameterChanged();
private:
    SimParameters* p_ = nullptr;
    SimConfDoubleSpinBox *cr_, *ld_, *wl_;
    QRadioButton *hexR_, *sqR_;
    SimConfIntSpinBox* gs_;
    SimConfCheckBox* ecm_;
};

class SimConfPageIon : public QScrollArea {
    Q_OBJECT
public:
    explicit SimConfPageIon(QWidget* parent = nullptr) : QScrollArea(parent) {
        setObjectName("page_ions"); setWidgetResizable(true);
        auto* c = new QWidget(this); auto* l = new QVBoxLayout(c);
        auto* g = new QGroupBox(tr("Ion Profile"), c);
        auto* fl = new QFormLayout(g);
        combo_ = new QComboBox(g);
        combo_->addItems({tr("Basic"), tr("Basic + Ca2+"), tr("Mammal"), tr("Amphibian"), tr("Custom")});
        fl->addRow(tr("Profile Type:"), combo_);
        l->addWidget(g);
        customGrp_ = new QGroupBox(tr("Custom Ion Settings"), c);
        customGrp_->setVisible(false);
        auto* cl = new QFormLayout(customGrp_);
        // Create labelled spin boxes for custom ion concentrations
        const char* ionNames[] = {"Na+", "K+", "Cl-", "Ca2+"};
        for (int i = 0; i < 4; i++) {
            ionSpin_[i] = new QDoubleSpinBox(customGrp_);
            ionSpin_[i]->setRange(0, 10000); ionSpin_[i]->setDecimals(4);
            cl->addRow(tr("%1 cell (mM):").arg(ionNames[i]), ionSpin_[i]);
            ionSpinEnv_[i] = new QDoubleSpinBox(customGrp_);
            ionSpinEnv_[i]->setRange(0, 10000); ionSpinEnv_[i]->setDecimals(4);
            cl->addRow(tr("%1 env (mM):").arg(ionNames[i]), ionSpinEnv_[i]);
        }
        l->addWidget(customGrp_); l->addStretch();
        setWidget(c);
        connect(combo_, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int i) {
            if (p_) { p_->ion_profile = static_cast<IonProfileType>(i);
                      customGrp_->setVisible(p_->ion_profile == IonProfileType::CUSTOM); }
        });
        // Ion spin box connections are deferred to init() when parameter pointer is available
    }
    void init(SimParameters* p, SimConfUndoStack*) {
        p_ = p;
        if (p_) {
            // Bind custom ion spin boxes to parameter storage
            connect(ionSpin_[0], QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { p_->custom_Na_cell = v; });
            connect(ionSpinEnv_[0], QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { p_->custom_Na_env = v; });
            connect(ionSpin_[1], QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { p_->custom_K_cell = v; });
            connect(ionSpinEnv_[1], QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { p_->custom_K_env = v; });
            connect(ionSpin_[2], QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { p_->custom_Cl_cell = v; });
            connect(ionSpinEnv_[2], QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { p_->custom_Cl_env = v; });
            connect(ionSpin_[3], QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { p_->custom_Ca_cell = v; });
            connect(ionSpinEnv_[3], QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { p_->custom_Ca_env = v; });
        }
        syncFromConfig();
    }
    void syncFromConfig() {
        if (!p_) return;
        combo_->blockSignals(true);
        combo_->setCurrentIndex(static_cast<int>(p_->ion_profile));
        combo_->blockSignals(false);
        customGrp_->setVisible(p_->ion_profile == IonProfileType::CUSTOM);
        // Sync custom ion values
        for (int i = 0; i < 4; i++) { ionSpin_[i]->blockSignals(true); ionSpinEnv_[i]->blockSignals(true); }
        ionSpin_[0]->setValue(p_->custom_Na_cell); ionSpinEnv_[0]->setValue(p_->custom_Na_env);
        ionSpin_[1]->setValue(p_->custom_K_cell);  ionSpinEnv_[1]->setValue(p_->custom_K_env);
        ionSpin_[2]->setValue(p_->custom_Cl_cell); ionSpinEnv_[2]->setValue(p_->custom_Cl_env);
        ionSpin_[3]->setValue(p_->custom_Ca_cell); ionSpinEnv_[3]->setValue(p_->custom_Ca_env);
        for (int i = 0; i < 4; i++) { ionSpin_[i]->blockSignals(false); ionSpinEnv_[i]->blockSignals(false); }
    }
private:
    SimParameters* p_ = nullptr;
    QComboBox* combo_;
    QGroupBox* customGrp_;
    QDoubleSpinBox* ionSpin_[4] = {};    // cell concentrations
    QDoubleSpinBox* ionSpinEnv_[4] = {}; // env concentrations
};

class SimConfPageTissueDefault : public QScrollArea {
    Q_OBJECT
public:
    explicit SimConfPageTissueDefault(QWidget* parent = nullptr) : QScrollArea(parent) {
        setObjectName("page_tissue_default"); setWidgetResizable(true);
        auto* c = new QWidget(this); auto* l = new QVBoxLayout(c);
        auto* gg = new QGroupBox(tr("Default Tissue Profile"), c);
        auto* gl = new QFormLayout(gg);
        name_ = new SimConfLineEdit(gg); imageFile_ = new SimConfFilePathEdit(gg);
        gl->addRow(tr("Name:"), name_); gl->addRow(tr("Image Mask:"), imageFile_);
        l->addWidget(gg);
        auto* mg = new QGroupBox(tr("Membrane Diffusion Constants (m^2/s)"), c);
        auto* ml = new QFormLayout(mg);
        const QStringList ions = {"Na","K","Cl","Ca","M","P"};
        for (const auto& ion : ions) {
            auto* s = new SimConfDoubleSpinBox(mg); s->setRange(0,1e-6); s->setDecimals(20);
            ml->addRow(tr("Dm_%1:").arg(ion), s); ionS_[ion.toStdString()] = s;
        }
        l->addWidget(mg); l->addStretch(); setWidget(c);
    }
    void init(SimParameters* p, SimConfUndoStack* u) {
        name_->initParam(p, &p->tissue_default.name, u, tr("Name"));
        imageFile_->initParam(p, &p->tissue_default.picker_image_filename, tr("Image"), tr("Images (*.png *.jpg *.bmp *.svg)"));
        ionS_["Na"]->initParam(p, &p->tissue_default.Dm_Na, u, tr("Dm_Na"));
        ionS_["K"]->initParam(p, &p->tissue_default.Dm_K, u, tr("Dm_K"));
        ionS_["Cl"]->initParam(p, &p->tissue_default.Dm_Cl, u, tr("Dm_Cl"));
        ionS_["Ca"]->initParam(p, &p->tissue_default.Dm_Ca, u, tr("Dm_Ca"));
        ionS_["M"]->initParam(p, &p->tissue_default.Dm_M, u, tr("Dm_M"));
        ionS_["P"]->initParam(p, &p->tissue_default.Dm_P, u, tr("Dm_P"));
    }
    void syncFromConfig() { name_->syncFromConfig(); imageFile_->syncFromConfig();
        for (auto& [k,s] : ionS_) s->syncFromConfig(); }
private:
    SimConfLineEdit* name_;
    SimConfFilePathEdit* imageFile_;
    std::map<std::string, SimConfDoubleSpinBox*> ionS_;
};

class SimConfPageTissueCustom : public QScrollArea {
    Q_OBJECT
public:
    explicit SimConfPageTissueCustom(QWidget* parent = nullptr) : QScrollArea(parent) {
        setObjectName("page_tissue_custom"); setWidgetResizable(true);
        auto* c = new QWidget(this); auto* l = new QVBoxLayout(c);
        auto* gg = new QGroupBox(tr("Custom Tissue Profile"), c);
        auto* gl = new QFormLayout(gg);
        name_ = new QLineEdit(gg); imageFile_ = new SimConfFilePathEdit(gg);
        gl->addRow(tr("Name:"), name_); gl->addRow(tr("Image Mask:"), imageFile_);
        l->addWidget(gg);
        auto* mg = new QGroupBox(tr("Membrane Diffusion Constants (m^2/s)"), c);
        auto* ml = new QFormLayout(mg);
        for (const auto& ion : {"Na","K","Cl","Ca","M","P"}) {
            auto* s = new QDoubleSpinBox(mg); s->setRange(0,1e-6); s->setDecimals(20);
            ml->addRow(tr("Dm_%1:").arg(ion), s); ionS_[ion] = s;
        }
        l->addWidget(mg); l->addStretch(); setWidget(c);
        connect(name_, &QLineEdit::editingFinished, this, [this]() {
            if (cur_) cur_->name = name_->text().toStdString();
        });
        connectSpinBoxes();
    }
    void showProfile(TissueProfile* p) {
        cur_ = p; if (!p) return;
        // Block signals during bulk update to prevent feedback loops
        name_->blockSignals(true);
        name_->setText(QString::fromStdString(p->name));
        name_->blockSignals(false);
        for (auto& [k, s] : ionS_) s->blockSignals(true);
        ionS_["Na"]->setValue(p->Dm_Na); ionS_["K"]->setValue(p->Dm_K);
        ionS_["Cl"]->setValue(p->Dm_Cl); ionS_["Ca"]->setValue(p->Dm_Ca);
        ionS_["M"]->setValue(p->Dm_M);   ionS_["P"]->setValue(p->Dm_P);
        for (auto& [k, s] : ionS_) s->blockSignals(false);
    }
private:
    void connectSpinBoxes() {
        // Connect each spin box to write back to the current TissueProfile
        auto connectIon = [this](const std::string& ion, double TissueProfile::* field) {
            connect(ionS_[ion], QOverload<double>::of(&QDoubleSpinBox::valueChanged),
                    this, [this, field](double v) {
                        if (cur_) cur_->*field = v;
                    });
        };
        connectIon("Na", &TissueProfile::Dm_Na);
        connectIon("K",  &TissueProfile::Dm_K);
        connectIon("Cl", &TissueProfile::Dm_Cl);
        connectIon("Ca", &TissueProfile::Dm_Ca);
        connectIon("M",  &TissueProfile::Dm_M);
        connectIon("P",  &TissueProfile::Dm_P);
    }
    TissueProfile* cur_ = nullptr;
    QLineEdit* name_;
    SimConfFilePathEdit* imageFile_;
    std::map<std::string, QDoubleSpinBox*> ionS_;
};

class SimConfPageExport : public QScrollArea {
    Q_OBJECT
public:
    explicit SimConfPageExport(QWidget* parent = nullptr) : QScrollArea(parent) {
        setObjectName("page_export"); setWidgetResizable(true);
        auto* c = new QWidget(this); auto* l = new QVBoxLayout(c);
        l->addWidget(new QLabel(tr(
            "Configure simulation export settings.\n\n"
            "Select a specific export category in the tree to configure its items.\n\n"
            "Export types:\n"
            "  - Animations: animated visualizations\n"
            "  - Plots: static plots\n"
            "  - CSV: data exports"), c));
        l->addStretch(); setWidget(c);
    }
};

class SimConfPageAnim : public QScrollArea {
    Q_OBJECT
public:
    explicit SimConfPageAnim(QWidget* parent = nullptr) : QScrollArea(parent) {
        setObjectName("page_anim"); setWidgetResizable(true);
        auto* c = new QWidget(this); auto* l = new QVBoxLayout(c);
        auto* g = new QGroupBox(tr("Animation Export Settings"), c);
        auto* fl = new QFormLayout(g);
        sas_ = new QCheckBox(tr("Show after sim"), g); sav_ = new QCheckBox(tr("Save after sim"), g);
        sws_ = new QCheckBox(tr("Show during sim"), g); swv_ = new QCheckBox(tr("Save during sim"), g);
        fl->addRow(sas_); fl->addRow(sav_); fl->addRow(sws_); fl->addRow(swv_);
        l->addWidget(g); l->addStretch(); setWidget(c);
    }
    void init(SimParameters* p) {
        p_ = p; syncFromConfig();
        connect(sas_, &QCheckBox::toggled, [this](bool v) { if (p_) p_->anim.is_after_sim_show = v; });
        connect(sav_, &QCheckBox::toggled, [this](bool v) { if (p_) p_->anim.is_after_sim_save = v; });
        connect(sws_, &QCheckBox::toggled, [this](bool v) { if (p_) p_->anim.is_while_sim_show = v; });
        connect(swv_, &QCheckBox::toggled, [this](bool v) { if (p_) p_->anim.is_while_sim_save = v; });
    }
    void syncFromConfig() {
        if (!p_) return;
        for (auto* w : {sas_,sav_,sws_,swv_}) w->blockSignals(true);
        sas_->setChecked(p_->anim.is_after_sim_show); sav_->setChecked(p_->anim.is_after_sim_save);
        sws_->setChecked(p_->anim.is_while_sim_show); swv_->setChecked(p_->anim.is_while_sim_save);
        for (auto* w : {sas_,sav_,sws_,swv_}) w->blockSignals(false);
    }
private:
    SimParameters* p_ = nullptr;
    QCheckBox *sas_, *sav_, *sws_, *swv_;
};

class SimConfPageExportItem : public QScrollArea {
    Q_OBJECT
public:
    explicit SimConfPageExportItem(const QString& label, QWidget* parent = nullptr)
        : QScrollArea(parent) {
        setWidgetResizable(true);
        auto* c = new QWidget(this); auto* l = new QVBoxLayout(c);
        auto* g = new QGroupBox(label, c);
        auto* fl = new QFormLayout(g);
        nameE_ = new QLineEdit(g); enabledC_ = new QCheckBox(tr("Enabled"), g);
        pipeC_ = new QComboBox(g);
        fl->addRow(tr("Name:"), nameE_); fl->addRow(enabledC_); fl->addRow(tr("Pipeline:"), pipeC_);
        l->addWidget(g); l->addStretch(); setWidget(c);
        connect(nameE_, &QLineEdit::editingFinished, [this]() { if (cur_) cur_->name = nameE_->text().toStdString(); });
        connect(enabledC_, &QCheckBox::toggled, [this](bool v) { if (cur_) cur_->is_enabled = v; });
    }
    void showItem(ExportItem* item, const QStringList& pipes) {
        cur_ = item; if (!item) return;
        nameE_->setText(QString::fromStdString(item->name));
        enabledC_->setChecked(item->is_enabled);
        pipeC_->blockSignals(true); pipeC_->clear(); pipeC_->addItems(pipes);
        int idx = pipes.indexOf(QString::fromStdString(item->pipeline_name));
        pipeC_->setCurrentIndex(idx >= 0 ? idx : 0); pipeC_->blockSignals(false);
    }
private:
    ExportItem* cur_ = nullptr;
    QLineEdit* nameE_; QCheckBox* enabledC_; QComboBox* pipeC_;
};

class SimConfPagePlot : public QScrollArea {
    Q_OBJECT
public:
    explicit SimConfPagePlot(QWidget* parent = nullptr) : QScrollArea(parent) {
        setObjectName("page_plot"); setWidgetResizable(true);
        auto* c = new QWidget(this); auto* l = new QVBoxLayout(c);
        auto* g = new QGroupBox(tr("Plot Export Settings"), c);
        auto* fl = new QFormLayout(g);
        sas_ = new QCheckBox(tr("Show plots after sim"), g);
        sav_ = new QCheckBox(tr("Save plots after sim"), g);
        fl->addRow(sas_); fl->addRow(sav_);
        l->addWidget(g); l->addStretch(); setWidget(c);
    }
    void init(SimParameters* p) {
        p_ = p; syncFromConfig();
        connect(sas_, &QCheckBox::toggled, [this](bool v) { if (p_) p_->plot.is_after_sim_show = v; });
        connect(sav_, &QCheckBox::toggled, [this](bool v) { if (p_) p_->plot.is_after_sim_save = v; });
    }
    void syncFromConfig() {
        if (!p_) return;
        sas_->blockSignals(true); sav_->blockSignals(true);
        sas_->setChecked(p_->plot.is_after_sim_show); sav_->setChecked(p_->plot.is_after_sim_save);
        sas_->blockSignals(false); sav_->blockSignals(false);
    }
private:
    SimParameters* p_ = nullptr;
    QCheckBox *sas_, *sav_;
};

class SimConfPageCSV : public QScrollArea {
    Q_OBJECT
public:
    explicit SimConfPageCSV(QWidget* parent = nullptr) : QScrollArea(parent) {
        setObjectName("page_csv"); setWidgetResizable(true);
        auto* c = new QWidget(this); auto* l = new QVBoxLayout(c);
        l->addWidget(new QLabel(tr("Select individual CSV export items in the tree."), c));
        l->addStretch(); setWidget(c);
    }
};

// ---------------------------------------------------------------------------
// SimConf Stacked Widget
// ---------------------------------------------------------------------------
class SimConfStack : public QStackedWidget {
    Q_OBJECT
public:
    explicit SimConfStack(QWidget* parent = nullptr) : QStackedWidget(parent) {
        setObjectName("sim_conf_stack");
        pPath_ = new SimConfPagePath(this);
        pTime_ = new SimConfPageTime(this);
        pSpace_ = new SimConfPageSpace(this);
        pIon_ = new SimConfPageIon(this);
        pTisDef_ = new SimConfPageTissueDefault(this);
        pTisCust_ = new SimConfPageTissueCustom(this);
        pExport_ = new SimConfPageExport(this);
        pAnim_ = new SimConfPageAnim(this);
        pAnimItem_ = new SimConfPageExportItem(tr("Cell Cluster Anim Export"), this);
        pPlot_ = new SimConfPagePlot(this);
        pPlotCellItem_ = new SimConfPageExportItem(tr("Single Cell Plot Export"), this);
        pPlotCellsItem_ = new SimConfPageExportItem(tr("Cell Cluster Plot Export"), this);
        pCsv_ = new SimConfPageCSV(this);
        pCsvItem_ = new SimConfPageExportItem(tr("CSV Data Export"), this);
        auto* phAnimCells = new QLabel(tr("Select an animation export item."), this);
        auto* phPlotCell = new QLabel(tr("Select a single cell plot item."), this);
        auto* phPlotCells = new QLabel(tr("Select a cell cluster plot item."), this);
        idx_["page_paths"] = addWidget(pPath_);
        idx_["page_time"] = addWidget(pTime_);
        idx_["page_space"] = addWidget(pSpace_);
        idx_["page_ions"] = addWidget(pIon_);
        idx_["page_tissue_default"] = addWidget(pTisDef_);
        idx_["page_tissue_custom"] = addWidget(pTisCust_);
        idx_["page_export"] = addWidget(pExport_);
        idx_["page_anim"] = addWidget(pAnim_);
        idx_["page_anim_cells"] = addWidget(phAnimCells);
        idx_["page_anim_cells_item"] = addWidget(pAnimItem_);
        idx_["page_plot"] = addWidget(pPlot_);
        idx_["page_plot_cell"] = addWidget(phPlotCell);
        idx_["page_plot_cell_item"] = addWidget(pPlotCellItem_);
        idx_["page_plot_cells"] = addWidget(phPlotCells);
        idx_["page_plot_cells_item"] = addWidget(pPlotCellsItem_);
        idx_["page_csv"] = addWidget(pCsv_);
        idx_["page_csv_item"] = addWidget(pCsvItem_);
    }
    void init(SimParameters* p, SimConfUndoStack* u) {
        p_ = p;
        pPath_->init(p, u); pTime_->init(p, u); pSpace_->init(p, u);
        pIon_->init(p, u); pTisDef_->init(p, u); pAnim_->init(p); pPlot_->init(p);
    }
    void syncAllFromConfig() {
        pPath_->syncFromConfig(); pTime_->syncFromConfig(); pSpace_->syncFromConfig();
        pIon_->syncFromConfig(); pTisDef_->syncFromConfig(); pAnim_->syncFromConfig(); pPlot_->syncFromConfig();
    }
public slots:
    void switchToPage(const QString& pg) {
        auto tryDynamic = [&](const QString& prefix, auto& vec, auto* page) -> bool {
            if (!pg.startsWith(prefix)) return false;
            bool ok; int i = pg.midRef(prefix.length()).toInt(&ok);
            if (ok && i >= 0 && i < static_cast<int>(vec.size())) {
                page->showItem(&vec[i], {}); setCurrentWidget(page);
            }
            return true;
        };
        if (pg.startsWith("page_tissue_custom_")) {
            bool ok; int i = pg.midRef(QString("page_tissue_custom_").length()).toInt(&ok);
            if (ok && p_ && i >= 0 && i < static_cast<int>(p_->tissue_custom.size())) {
                pTisCust_->showProfile(&p_->tissue_custom[i]); setCurrentWidget(pTisCust_);
            }
            return;
        }
        if (p_) {
            if (tryDynamic("page_anim_cells_item_", p_->anim.anims_after_sim, pAnimItem_)) return;
            if (tryDynamic("page_plot_cell_item_", p_->plot.plots_cell_after_sim, pPlotCellItem_)) return;
            if (tryDynamic("page_plot_cells_item_", p_->plot.plots_cells_after_sim, pPlotCellsItem_)) return;
            if (tryDynamic("page_csv_item_", p_->csv.csvs_after_sim, pCsvItem_)) return;
        }
        auto it = idx_.find(pg);
        if (it != idx_.end()) setCurrentIndex(it->second);
    }
private:
    SimParameters* p_ = nullptr;
    SimConfPagePath* pPath_; SimConfPageTime* pTime_; SimConfPageSpace* pSpace_;
    SimConfPageIon* pIon_; SimConfPageTissueDefault* pTisDef_; SimConfPageTissueCustom* pTisCust_;
    SimConfPageExport* pExport_; SimConfPageAnim* pAnim_;
    SimConfPageExportItem* pAnimItem_; SimConfPagePlot* pPlot_;
    SimConfPageExportItem* pPlotCellItem_; SimConfPageExportItem* pPlotCellsItem_;
    SimConfPageCSV* pCsv_; SimConfPageExportItem* pCsvItem_;
    std::map<QString, int> idx_;
};

// ---------------------------------------------------------------------------
// Simulation Configurator (port of gui/simconf/guisimconf.py)
// ---------------------------------------------------------------------------
class SimConf : public QObject {
    Q_OBJECT
public:
    explicit SimConf(QObject* parent = nullptr) : QObject(parent),
        params_(std::make_unique<SimParameters>()), undoStack_(new SimConfUndoStack(this)) {}
    SimParameters* params() { return params_.get(); }
    SimConfUndoStack* undoStack() { return undoStack_; }
    bool isOpen() const { return params_->is_loaded; }
    bool isDirty() const { return dirty_; }
    void setDirty(bool d) { if (d && !isOpen()) return; dirty_ = d; emit dirtyChanged(dirty_); }
    bool save() {
        if (!isOpen()) return false;
        if (params_->save()) { setDirty(false); emit statusMessage(tr("Configuration saved.")); return true; }
        return false;
    }
    bool saveAs(const QString& fn) {
        if (params_->save(fn.toStdString())) { setDirty(false); emit filenameChanged(fn);
            emit statusMessage(tr("Saved as %1.").arg(fn)); return true; }
        return false;
    }
    bool load(const QString& fn) {
        if (params_->load(fn.toStdString())) { undoStack_->clear(); undoStack_->setSimConfOpen(true);
            setDirty(false); emit filenameChanged(fn); emit configLoaded();
            emit statusMessage(tr("Loaded: %1").arg(fn)); return true; }
        return false;
    }
    void unload() { params_->unload(); undoStack_->clear(); undoStack_->setSimConfOpen(false);
        dirty_ = false; emit filenameChanged(QString()); emit configUnloaded(); }
    bool saveIfDirty() {
        if (!isOpen() || !dirty_) return true;
        auto r = QMessageBox::question(nullptr, tr("Save?"),
            tr("Unsaved changes exist. Save?"),
            QMessageBox::Save | QMessageBox::Discard | QMessageBox::Cancel, QMessageBox::Save);
        if (r == QMessageBox::Save) return save();
        if (r == QMessageBox::Cancel) return false;
        return true;
    }
signals:
    void filenameChanged(const QString& fn);
    void dirtyChanged(bool dirty);
    void configLoaded();
    void configUnloaded();
    void statusMessage(const QString& msg);
private:
    std::unique_ptr<SimParameters> params_;
    SimConfUndoStack* undoStack_;
    bool dirty_ = false;
};

// ---------------------------------------------------------------------------
// Simulator Phase Controller
// ---------------------------------------------------------------------------
class SimmerPhase : public QObject {
    Q_OBJECT
public:
    explicit SimmerPhase(SimPhaseKind kind, QObject* parent = nullptr)
        : QObject(parent), kind_(kind) {}
    SimPhaseKind kind() const { return kind_; }
    SimmerState state() const { return state_; }
    bool isQueued() const { return qModel_ || qExport_; }
    bool isQueuedModelling() const { return qModel_; }
    bool isQueuedExporting() const { return qExport_; }
    void setState(SimmerState s) { auto old = state_; state_ = s; emit stateChanged(state_, old); }
    QString name() const {
        auto it = SIM_PHASE_KIND_TO_NAME.find(kind_);
        return it != SIM_PHASE_KIND_TO_NAME.end() ? QString::fromStdString(it->second) : "unknown";
    }
    QString statusText() const {
        auto it = SIMMER_STATE_TO_PHASE_STATUS.find(state_);
        return it != SIMMER_STATE_TO_PHASE_STATUS.end() ? QString::fromStdString(it->second) : "";
    }
signals:
    void stateChanged(SimmerState, SimmerState);
    void queueChanged();
public slots:
    void setQueuedModelling(bool v) { qModel_ = v; upd(); emit queueChanged(); }
    void setQueuedExporting(bool v) { qExport_ = v; upd(); emit queueChanged(); }
private:
    void upd() { if (SIMMER_STATES_IDLE.count(state_)) setState(isQueued() ? SimmerState::QUEUED : SimmerState::UNQUEUED); }
    SimPhaseKind kind_;
    SimmerState state_ = SimmerState::UNQUEUED;
    bool qModel_ = true, qExport_ = false;
};

// ---------------------------------------------------------------------------
// Simulator Proactor
// ---------------------------------------------------------------------------
class SimmerProactor : public QObject {
    Q_OBJECT
public:
    explicit SimmerProactor(QObject* parent = nullptr) : QObject(parent) {
        pSeed_ = new SimmerPhase(SimPhaseKind::SEED, this);
        pInit_ = new SimmerPhase(SimPhaseKind::INIT, this);
        pSim_  = new SimmerPhase(SimPhaseKind::SIM, this);
    }
    SimmerState state() const { return state_; }
    SimmerPhase* phaseSeed() { return pSeed_; }
    SimmerPhase* phaseInit() { return pInit_; }
    SimmerPhase* phaseSim()  { return pSim_; }
    bool isQueued() const { return pSeed_->isQueued() || pInit_->isQueued() || pSim_->isQueued(); }
    bool isRunning() const { return SIMMER_STATES_RUNNING.count(state_) > 0; }
    bool isWorking() const { return SIMMER_STATES_WORKING.count(state_) > 0; }
    bool isWorkable() const { return SIMMER_STATES_UNWORKABLE.count(state_) == 0 && isQueued(); }

    void init(SimConf* sc) {
        sc_ = sc; setState(SimmerState::UNQUEUED);
        for (auto* ph : {pSeed_, pInit_, pSim_})
            connect(ph, &SimmerPhase::queueChanged, this, [this]() {
                if (SIMMER_STATES_IDLE.count(state_))
                    setState(isQueued() ? SimmerState::QUEUED : SimmerState::UNQUEUED);
            });
    }
    void haltWorkers() {
        if (!isWorking()) return;
        stopWorkers();
        if (proc_ && proc_->state() != QProcess::NotRunning) proc_->kill();
    }
public slots:
    void toggleWork() {
        if (isRunning()) {
            setState(SimmerState::PAUSED);
        } else if (state_ == SimmerState::PAUSED) {
            setState(prevRun_);
        } else if (isWorkable()) {
            startWork();
        }
    }
    void stopWorkers() {
        if (!isWorking()) return;
        setState(SimmerState::STOPPING);
        if (proc_) proc_->terminate();
        wq_.clear();
    }
signals:
    void stateChanged(SimmerState, SimmerState);
    void progressRangeChanged(int, int);
    void progressValueChanged(int);
    void progressTextChanged(const QString&);
private:
    void setState(SimmerState s) {
        auto old = state_; state_ = s;
        if (SIMMER_STATES_RUNNING.count(s)) prevRun_ = s;
        emit stateChanged(state_, old);
    }
    void startWork() {
        if (!sc_ || !sc_->isOpen()) return;
        wq_.clear();
        for (auto* ph : {pSeed_, pInit_, pSim_}) {
            if (ph->isQueuedModelling()) wq_.push_back({ph, SimmerPhaseSubkind::MODELLING});
            if (ph->isQueuedExporting()) wq_.push_back({ph, SimmerPhaseSubkind::EXPORTING});
        }
        if (wq_.empty()) return;
        runNext();
    }
    void runNext() {
        if (wq_.empty()) {
            setState(SimmerState::FINISHED);
            for (auto* ph : {pSeed_, pInit_, pSim_}) ph->setState(SimmerState::FINISHED);
            return;
        }
        auto [ph, sk] = wq_.front(); wq_.pop_front(); curPh_ = ph;
        auto rs = sk == SimmerPhaseSubkind::MODELLING ? SimmerState::MODELLING : SimmerState::EXPORTING;
        setState(rs); ph->setState(rs);
        proc_ = new QProcess(this);
        connect(proc_, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), this, &SimmerProactor::onDone);
        connect(proc_, &QProcess::readyReadStandardOutput, this, &SimmerProactor::onOutput);
        connect(proc_, &QProcess::readyReadStandardError, this, &SimmerProactor::onOutput);
        QString cf = QString::fromStdString(sc_->params()->conf_filename);
        QStringList args;
        if (sk == SimmerPhaseSubkind::MODELLING) {
            if (ph->kind() == SimPhaseKind::SEED) args << "seed";
            else if (ph->kind() == SimPhaseKind::INIT) args << "init";
            else args << "sim";
        } else {
            if (ph->kind() == SimPhaseKind::SEED) args << "plot" << "seed";
            else if (ph->kind() == SimPhaseKind::INIT) args << "plot" << "init";
            else args << "plot" << "sim";
        }
        args << cf;
        // Validate config file exists before launching
        if (cf.isEmpty()) {
            emit progressTextChanged(tr("Error: No configuration file set. Save your config first."));
            setState(SimmerState::FINISHED);
            return;
        }
        if (!QFileInfo::exists(cf)) {
            emit progressTextChanged(tr("Error: Config file not found: %1").arg(cf));
            setState(SimmerState::FINISHED);
            return;
        }
        // Use betse from PATH, or try common locations
        QString betseBin = "betse";
        QStringList searchPaths = {
            QDir::homePath() + "/Documents/biotools/.venv/bin/betse",
            "/usr/local/bin/betse",
            "/usr/bin/betse"
        };
        for (const auto& p : searchPaths) {
            if (QFileInfo::exists(p)) { betseBin = p; break; }
        }
        emit progressTextChanged(tr("Running BETSE %1: %2 %3").arg(ph->name(), betseBin, args.join(" ")));
        proc_->start(betseBin, args);
    }
private slots:
    void onDone(int exitCode, QProcess::ExitStatus es) {
        if (es == QProcess::CrashExit || exitCode != 0)
            emit progressTextChanged(tr("BETSE failed (exit %1).").arg(exitCode));
        if (curPh_) curPh_->setState(SimmerState::FINISHED);
        proc_->deleteLater(); proc_ = nullptr;
        if (state_ == SimmerState::STOPPING) { setState(SimmerState::FINISHED); wq_.clear(); }
        else runNext();
    }
    void onOutput() {
        if (!proc_) return;
        QString out = proc_->readAllStandardOutput() + proc_->readAllStandardError();
        QRegularExpression re(R"((?:step|Step)\s+(\d+)\s+(?:of|OF)\s+(\d+))");
        auto m = re.match(out);
        if (m.hasMatch()) { emit progressRangeChanged(0, m.captured(2).toInt()); emit progressValueChanged(m.captured(1).toInt()); }
        emit progressTextChanged(out.trimmed().split('\n').last());
    }
private:
    SimConf* sc_ = nullptr;
    SimmerState state_ = SimmerState::UNQUEUED, prevRun_ = SimmerState::MODELLING;
    SimmerPhase *pSeed_, *pInit_, *pSim_, *curPh_ = nullptr;
    QProcess* proc_ = nullptr;
    struct WI { SimmerPhase* ph; SimmerPhaseSubkind sk; };
    std::deque<WI> wq_;
};

// ---------------------------------------------------------------------------
// Simulator Tab Widget
// ---------------------------------------------------------------------------
class SimmerTabWidget : public QTabWidget {
    Q_OBJECT
public:
    explicit SimmerTabWidget(QWidget* parent = nullptr) : QTabWidget(parent) {
        setObjectName("sim_tab");
        auto* rt = new QWidget(this); auto* rl = new QVBoxLayout(rt);
        auto* qg = new QGroupBox(tr("Simulation Phase Queue"), rt);
        auto* gl = new QGridLayout(qg);
        gl->addWidget(new QLabel(tr("Phase")), 0, 0);
        gl->addWidget(new QLabel(tr("Model")), 0, 1);
        gl->addWidget(new QLabel(tr("Lock")), 0, 2);
        gl->addWidget(new QLabel(tr("Export")), 0, 3);
        gl->addWidget(new QLabel(tr("Status")), 0, 4);
        gl->addWidget(new QLabel(tr("Seed")), 1, 0);
        seedM_ = new QCheckBox(rt); seedM_->setChecked(true); seedL_ = new QCheckBox(rt);
        seedS_ = new QLabel(tr("Unqueued"), rt);
        gl->addWidget(seedM_,1,1); gl->addWidget(seedL_,1,2); gl->addWidget(new QLabel(tr("N/A")),1,3); gl->addWidget(seedS_,1,4);
        gl->addWidget(new QLabel(tr("Init")), 2, 0);
        initM_ = new QCheckBox(rt); initM_->setChecked(true); initL_ = new QCheckBox(rt); initE_ = new QCheckBox(rt);
        initS_ = new QLabel(tr("Unqueued"), rt);
        gl->addWidget(initM_,2,1); gl->addWidget(initL_,2,2); gl->addWidget(initE_,2,3); gl->addWidget(initS_,2,4);
        gl->addWidget(new QLabel(tr("Sim")), 3, 0);
        simM_ = new QCheckBox(rt); simM_->setChecked(true); simL_ = new QCheckBox(rt); simE_ = new QCheckBox(rt);
        simS_ = new QLabel(tr("Unqueued"), rt);
        gl->addWidget(simM_,3,1); gl->addWidget(simL_,3,2); gl->addWidget(simE_,3,3); gl->addWidget(simS_,3,4);
        rl->addWidget(qg);
        ptf_ = new QFrame(rt); auto* pl = new QHBoxLayout(ptf_);
        toggleBtn_ = new QPushButton(tr("Start"), ptf_); toggleBtn_->setCheckable(true);
        stopBtn_ = new QPushButton(tr("Stop"), ptf_);
        pl->addWidget(toggleBtn_); pl->addWidget(stopBtn_); pl->addStretch();
        rl->addWidget(ptf_);
        pStatus_ = new QLabel(tr("Waiting for phase(s) to be queued..."), rt);
        pBar_ = new QProgressBar(rt); pBar_->setRange(0,100); pBar_->setValue(0);
        pSub_ = new QLabel(rt); pSubG_ = new QFrame(rt);
        auto* sl = new QVBoxLayout(pSubG_); sl->addWidget(pSub_);
        rl->addWidget(pStatus_); rl->addWidget(pBar_); rl->addWidget(pSubG_); rl->addStretch();
        addTab(rt, tr("Simulator"));
    }
    void init(SimConf* sc) {
        pro_ = new SimmerProactor(this); pro_->init(sc);
        connect(seedM_, &QCheckBox::toggled, pro_->phaseSeed(), &SimmerPhase::setQueuedModelling);
        connect(initM_, &QCheckBox::toggled, pro_->phaseInit(), &SimmerPhase::setQueuedModelling);
        connect(initE_, &QCheckBox::toggled, pro_->phaseInit(), &SimmerPhase::setQueuedExporting);
        connect(simM_, &QCheckBox::toggled, pro_->phaseSim(), &SimmerPhase::setQueuedModelling);
        connect(simE_, &QCheckBox::toggled, pro_->phaseSim(), &SimmerPhase::setQueuedExporting);
        connect(toggleBtn_, &QPushButton::clicked, pro_, &SimmerProactor::toggleWork);
        connect(stopBtn_, &QPushButton::clicked, pro_, &SimmerProactor::stopWorkers);
        connect(pro_, &SimmerProactor::stateChanged, this, &SimmerTabWidget::onState);
        connect(pro_, &SimmerProactor::progressRangeChanged, pBar_, &QProgressBar::setRange);
        connect(pro_, &SimmerProactor::progressValueChanged, pBar_, &QProgressBar::setValue);
        connect(pro_, &SimmerProactor::progressTextChanged, pSub_, &QLabel::setText);
        connect(pro_->phaseSeed(), &SimmerPhase::stateChanged, [this]() { seedS_->setText(pro_->phaseSeed()->statusText()); });
        connect(pro_->phaseInit(), &SimmerPhase::stateChanged, [this]() { initS_->setText(pro_->phaseInit()->statusText()); });
        connect(pro_->phaseSim(), &SimmerPhase::stateChanged, [this]() { simS_->setText(pro_->phaseSim()->statusText()); });
    }
    void haltWork() { if (pro_) pro_->haltWorkers(); }
private slots:
    void onState(SimmerState ns, SimmerState) {
        ptf_->setEnabled(pro_->isQueued());
        toggleBtn_->setEnabled(pro_->isWorkable());
        toggleBtn_->setChecked(pro_->isRunning());
        stopBtn_->setEnabled(pro_->isWorking());
        auto it = SIMMER_STATE_TO_PROACTOR_STATUS.find(ns);
        if (it != SIMMER_STATE_TO_PROACTOR_STATUS.end()) pStatus_->setText(QString::fromStdString(it->second));
        if (SIMMER_STATES_IDLE.count(ns) || ns == SimmerState::MODELLING) pBar_->reset();
        toggleBtn_->setText(pro_->isRunning() ? tr("Pause") : (ns == SimmerState::PAUSED ? tr("Resume") : tr("Start")));
    }
private:
    SimmerProactor* pro_ = nullptr;
    QFrame* ptf_; QPushButton *toggleBtn_, *stopBtn_;
    QProgressBar* pBar_; QLabel *pStatus_, *pSub_; QFrame* pSubG_;
    QCheckBox *seedM_, *seedL_, *initM_, *initL_, *initE_, *simM_, *simL_, *simE_;
    QLabel *seedS_, *initS_, *simS_;
};

// ---------------------------------------------------------------------------
// About Dialog
// ---------------------------------------------------------------------------
class AboutDialog : public QDialog {
    Q_OBJECT
public:
    explicit AboutDialog(QWidget* parent = nullptr) : QDialog(parent) {
        setWindowTitle(tr("About BETSEE")); setMinimumSize(500, 400);
        auto* l = new QVBoxLayout(this);
        auto* t = new QLabel(QString("<h2>%1 %2</h2>").arg(AppMetadata::NAME, AppMetadata::VERSION), this);
        t->setAlignment(Qt::AlignCenter); l->addWidget(t);
        auto* d = new QLabel(AppMetadata::DESCRIPTION, this); d->setWordWrap(true); l->addWidget(d);
        l->addSpacing(10);
        l->addWidget(new QLabel(tr("<b>Authors:</b> %1").arg(AppMetadata::AUTHORS), this));
        l->addWidget(new QLabel(tr("<b>License:</b> %1").arg(AppMetadata::LICENSE), this));
        auto* u = new QLabel(tr("<b>Homepage:</b> <a href=\"%1\">%1</a>").arg(AppMetadata::URL_HOMEPAGE), this);
        u->setOpenExternalLinks(true); l->addWidget(u); l->addStretch();
        auto* bb = new QDialogButtonBox(QDialogButtonBox::Ok, this);
        connect(bb, &QDialogButtonBox::accepted, this, &QDialog::accept); l->addWidget(bb);
    }
};

// ---------------------------------------------------------------------------
// Preferences Dialog
// ---------------------------------------------------------------------------
class PreferencesDialog : public QDialog {
    Q_OBJECT
public:
    explicit PreferencesDialog(QWidget* parent = nullptr) : QDialog(parent) {
        setWindowTitle(tr("Preferences")); setMinimumSize(500, 400);
        auto* l = new QVBoxLayout(this);
        auto* tw = new QTabWidget(this);
        auto* gt = new QWidget(tw); auto* gl = new QFormLayout(gt);
        gl->addRow(tr("BETSE executable:"), new QLineEdit("betse", gt));
        gl->addRow(tr("Default config dir:"),
            new QLineEdit(QStandardPaths::writableLocation(QStandardPaths::HomeLocation), gt));
        auto* lc = new QComboBox(gt); lc->addItems({"DEBUG","INFO","WARNING","ERROR"}); lc->setCurrentIndex(1);
        gl->addRow(tr("Log level:"), lc);
        tw->addTab(gt, tr("General"));
        auto* dt = new QWidget(tw); auto* dl = new QFormLayout(dt);
        auto* tc = new QComboBox(dt); tc->addItems({"System Default","Light","Dark"});
        dl->addRow(tr("Theme:"), tc);
        auto* fs = new QSpinBox(dt); fs->setRange(6,72); fs->setValue(10);
        dl->addRow(tr("Font size:"), fs);
        tw->addTab(dt, tr("Display"));
        l->addWidget(tw);
        auto* bb = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, this);
        connect(bb, &QDialogButtonBox::accepted, this, &QDialog::accept);
        connect(bb, &QDialogButtonBox::rejected, this, &QDialog::reject);
        l->addWidget(bb);
    }
};

// ---------------------------------------------------------------------------
// New Simulation Dialog
// ---------------------------------------------------------------------------
class NewSimDialog : public QDialog {
    Q_OBJECT
public:
    explicit NewSimDialog(QWidget* parent = nullptr) : QDialog(parent) {
        setWindowTitle(tr("New Simulation Configuration")); setMinimumSize(500, 200);
        auto* l = new QVBoxLayout(this);
        l->addWidget(new QLabel(tr("Create a new simulation configuration.\n"
            "Choose a location and filename."), this));
        auto* fl = new QHBoxLayout();
        fileE_ = new QLineEdit(this); fileE_->setPlaceholderText(tr("sim_config.yaml"));
        auto* bb = new QPushButton(tr("Browse..."), this);
        fl->addWidget(new QLabel(tr("File:"), this)); fl->addWidget(fileE_, 1); fl->addWidget(bb);
        l->addLayout(fl);
        connect(bb, &QPushButton::clicked, [this]() {
            QString f = QFileDialog::getSaveFileName(this, tr("New Config"),
                fileE_->text().isEmpty() ? "sim_config.yaml" : fileE_->text(),
                tr("YAML Files (*.yaml *.yml)"));
            if (!f.isEmpty()) fileE_->setText(f);
        });
        l->addStretch();
        auto* db = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, this);
        connect(db, &QDialogButtonBox::accepted, this, &QDialog::accept);
        connect(db, &QDialogButtonBox::rejected, this, &QDialog::reject);
        l->addWidget(db);
    }
    QString filename() const { return fileE_->text(); }
private:
    QLineEdit* fileE_;
};

// ---------------------------------------------------------------------------
// Main Window (port of gui/window/guiwindow.py + gui/guimain.py)
// ---------------------------------------------------------------------------
class BetseeMainWindow : public QMainWindow {
    Q_OBJECT
public:
    explicit BetseeMainWindow(const QString& initFile = QString(), QWidget* parent = nullptr)
        : QMainWindow(parent) {
        setWindowTitle(AppMetadata::NAME);
        setMinimumSize(1100, 750); resize(1280, 800);

        signaler_ = new BetseeSignaler(this);
        settings_ = new BetseeSettings(this);
        simConf_ = new SimConf(this);

        createActions(); createMenus(); createToolbar(); createStatusBar();
        createCentralWidget(); createDockWidgets();

        simConf_->undoStack()->init(this);
        connectSignals();

        tree_->init(simConf_->params());
        stack_->init(simConf_->params(), simConf_->undoStack());
        simTab_->init(simConf_);

        if (!initFile.isEmpty()) simConf_->load(initFile);
        else setSimConfOpen(false);

        signaler_->restoreSettingsSignal();
        for (auto* b : toolbar_->findChildren<QToolButton*>()) b->setFocusPolicy(Qt::NoFocus);
    }

    SimConf* simConf() { return simConf_; }
    LogViewer* logViewer() { return log_; }

protected:
    void closeEvent(QCloseEvent* ev) override {
        if (simConf_->saveIfDirty()) {
            simTab_->haltWork(); signaler_->storeSettingsSignal(); ev->accept();
        } else ev->ignore();
    }

private:
    BetseeSignaler* signaler_; BetseeSettings* settings_; SimConf* simConf_;
    QToolBar* toolbar_; LogViewer* log_;
    SimConfTree* tree_; SimConfStack* stack_; SimmerTabWidget* simTab_;
    QAction *aNew_, *aOpen_, *aClose_, *aSave_, *aSaveAs_, *aExit_;
    QAction *aPrefs_, *aAboutE_, *aAboutEE_;
    QAction *aTreeAdd_, *aTreeRm_, *aToggle_, *aStop_;
    QMenu *mFile_, *mEdit_, *mView_, *mHelp_;

    void createActions() {
        aNew_ = new QAction(tr("&New Simulation..."), this); aNew_->setShortcuts(QKeySequence::New);
        aOpen_ = new QAction(tr("&Open Simulation..."), this); aOpen_->setShortcuts(QKeySequence::Open);
        aClose_ = new QAction(tr("&Close Simulation"), this); aClose_->setShortcuts(QKeySequence::Close); aClose_->setEnabled(false);
        aSave_ = new QAction(tr("&Save Simulation"), this); aSave_->setShortcuts(QKeySequence::Save); aSave_->setEnabled(false);
        aSaveAs_ = new QAction(tr("Save Simulation &As..."), this); aSaveAs_->setShortcuts(QKeySequence::SaveAs); aSaveAs_->setEnabled(false);
        aExit_ = new QAction(tr("E&xit"), this); aExit_->setShortcuts(QKeySequence::Quit);
        aPrefs_ = new QAction(tr("&Preferences..."), this);
        aAboutE_ = new QAction(tr("About &BETSE..."), this);
        aAboutEE_ = new QAction(tr("About B&ETSEE..."), this);
        aTreeAdd_ = new QAction(tr("Add Item"), this);
        aTreeRm_ = new QAction(tr("Remove Item"), this);
        aToggle_ = new QAction(tr("Start/Pause"), this); aToggle_->setCheckable(true);
        aStop_ = new QAction(tr("Stop"), this);
    }
    void createMenus() {
        mFile_ = menuBar()->addMenu(tr("&File"));
        mFile_->addAction(aNew_); mFile_->addAction(aOpen_); mFile_->addSeparator();
        mFile_->addAction(aClose_); mFile_->addSeparator();
        mFile_->addAction(aSave_); mFile_->addAction(aSaveAs_); mFile_->addSeparator();
        mFile_->addAction(aExit_);
        mEdit_ = menuBar()->addMenu(tr("&Edit")); mEdit_->addSeparator(); mEdit_->addAction(aPrefs_);
        mView_ = menuBar()->addMenu(tr("&View"));
        mHelp_ = menuBar()->addMenu(tr("&Help")); mHelp_->addAction(aAboutE_); mHelp_->addAction(aAboutEE_);
    }
    void createToolbar() {
        toolbar_ = addToolBar(tr("Main Toolbar")); toolbar_->setObjectName("toolbar"); toolbar_->setMovable(false);
        toolbar_->addAction(aNew_); toolbar_->addAction(aOpen_); toolbar_->addAction(aSave_);
        toolbar_->addSeparator(); toolbar_->addSeparator();
        toolbar_->addAction(aTreeAdd_); toolbar_->addAction(aTreeRm_);
    }
    void createStatusBar() { statusBar()->showMessage(tr("Ready")); }
    void createCentralWidget() {
        auto* cs = new QSplitter(Qt::Horizontal, this);
        auto* tf = new QFrame(cs); tf->setObjectName("sim_conf_tree_frame");
        auto* tl = new QVBoxLayout(tf); tl->setContentsMargins(0,0,0,0);
        tree_ = new SimConfTree(tf); tl->addWidget(tree_);
        auto* tb = new QHBoxLayout();
        auto* ab = new QToolButton(tf); ab->setDefaultAction(aTreeAdd_); ab->setText("+");
        auto* rb = new QToolButton(tf); rb->setDefaultAction(aTreeRm_); rb->setText("-");
        tb->addWidget(ab); tb->addWidget(rb); tb->addStretch(); tl->addLayout(tb);
        cs->addWidget(tf);
        auto* rs = new QSplitter(Qt::Vertical, cs);
        stack_ = new SimConfStack(rs); rs->addWidget(stack_);
        simTab_ = new SimmerTabWidget(rs); rs->addWidget(simTab_);
        rs->setStretchFactor(0, 3); rs->setStretchFactor(1, 1);
        cs->addWidget(rs); cs->setStretchFactor(0, 1); cs->setStretchFactor(1, 4);
        setCentralWidget(cs);
    }
    void createDockWidgets() {
        auto* ld = new QDockWidget(tr("Log"), this); ld->setObjectName("log_dock");
        log_ = new LogViewer(ld); ld->setWidget(log_);
        addDockWidget(Qt::BottomDockWidgetArea, ld);
        mView_->addAction(ld->toggleViewAction());
    }
    void connectSignals() {
        connect(aNew_, &QAction::triggered, this, &BetseeMainWindow::onNew);
        connect(aOpen_, &QAction::triggered, this, &BetseeMainWindow::onOpen);
        connect(aClose_, &QAction::triggered, this, &BetseeMainWindow::onClose);
        connect(aSave_, &QAction::triggered, this, &BetseeMainWindow::onSave);
        connect(aSaveAs_, &QAction::triggered, this, &BetseeMainWindow::onSaveAs);
        connect(aExit_, &QAction::triggered, this, &QMainWindow::close);
        connect(aPrefs_, &QAction::triggered, this, [this]() { PreferencesDialog d(this); d.exec(); });
        connect(aAboutE_, &QAction::triggered, this, [this]() {
            QMessageBox::about(this, tr("About BETSE"),
                tr("<h3>BETSE</h3><p>BioElectric Tissue Simulation Engine</p>"
                   "<p><a href=\"https://github.com/betsee/betse\">github.com/betsee/betse</a></p>"));
        });
        connect(aAboutEE_, &QAction::triggered, this, [this]() { AboutDialog d(this); d.exec(); });
        connect(simConf_, &SimConf::filenameChanged, this, &BetseeMainWindow::onFnChanged);
        connect(simConf_, &SimConf::dirtyChanged, this, &BetseeMainWindow::onDirty);
        connect(simConf_, &SimConf::configLoaded, this, &BetseeMainWindow::onLoaded);
        connect(simConf_, &SimConf::configUnloaded, this, [this]() { setSimConfOpen(false); });
        connect(simConf_, &SimConf::statusMessage, this, [this](const QString& msg) { statusBar()->showMessage(msg); });
        connect(tree_, &SimConfTree::pageChangeRequested, stack_, &SimConfStack::switchToPage);
        connect(signaler_, &BetseeSignaler::restoreSettingsSignal, settings_, &BetseeSettings::restoreSettings);
        connect(signaler_, &BetseeSignaler::storeSettingsSignal, settings_, &BetseeSettings::storeSettings);
        // Connect all parameter-change signals to mark dirty
        auto cd = [this](QWidget* root) {
            for (auto* w : root->findChildren<SimConfDoubleSpinBox*>())
                connect(w, &SimConfDoubleSpinBox::parameterChanged, [this]() { simConf_->setDirty(true); });
            for (auto* w : root->findChildren<SimConfIntSpinBox*>())
                connect(w, &SimConfIntSpinBox::parameterChanged, [this]() { simConf_->setDirty(true); });
            for (auto* w : root->findChildren<SimConfCheckBox*>())
                connect(w, &SimConfCheckBox::parameterChanged, [this]() { simConf_->setDirty(true); });
            for (auto* w : root->findChildren<SimConfLineEdit*>())
                connect(w, &SimConfLineEdit::parameterChanged, [this]() { simConf_->setDirty(true); });
        };
        cd(stack_);
    }
    void setSimConfOpen(bool o) {
        aClose_->setEnabled(o); aSaveAs_->setEnabled(o); aSave_->setEnabled(false);
        stack_->setEnabled(o); tree_->setSimConfOpen(o); simTab_->setEnabled(o);
    }

private slots:
    void onNew() {
        NewSimDialog d(this);
        if (d.exec() == QDialog::Accepted && !d.filename().isEmpty()) {
            onClose();
            QString fn = d.filename();
            if (!fn.endsWith(".yaml") && !fn.endsWith(".yml")) fn += ".yaml";
            std::ofstream ofs(fn.toStdString());
            if (ofs.is_open()) {
                ofs << "# BETSE simulation configuration\n# Generated by BETSEE " << AppMetadata::VERSION << "\n\n"
                    << "general:\n    comp grid size: 50\n    simulate extracellular spaces: false\n"
                    << "    world length: 200.0e-6\n    ion profile: basic\n\n"
                    << "init time settings:\n    total time: 10.0\n    time step: 1.0e-3\n    sampling rate: 1.0\n\n"
                    << "sim time settings:\n    total time: 100.0\n    time step: 1.0e-3\n    sampling rate: 1.0\n\n"
                    << "cell cluster:\n    cell radius: 5.0e-6\n    lattice type: hex\n    lattice disorder: 0.4\n";
                ofs.close();
            }
            simConf_->load(fn);
            log_->logInfo(tr("Created: %1").arg(fn));
        }
    }
    void onOpen() {
        if (!simConf_->saveIfDirty()) return;
        QString fn = QFileDialog::getOpenFileName(this, tr("Open Config"), QString(), tr("YAML (*.yaml *.yml);;All (*)"));
        if (fn.isEmpty()) return;
        onClose();
        if (simConf_->load(fn)) log_->logInfo(tr("Opened: %1").arg(fn));
        else QMessageBox::critical(this, tr("Open Failed"), tr("Failed to open:\n%1").arg(fn));
    }
    void onClose() { if (!simConf_->saveIfDirty()) return; simConf_->unload(); setSimConfOpen(false); log_->logInfo(tr("Closed.")); }
    void onSave() { if (simConf_->save()) log_->logInfo(tr("Saved.")); }
    void onSaveAs() {
        QString fn = QFileDialog::getSaveFileName(this, tr("Save As"), QString(), tr("YAML (*.yaml *.yml)"));
        if (fn.isEmpty()) return;
        if (!fn.endsWith(".yaml") && !fn.endsWith(".yml")) fn += ".yaml";
        if (simConf_->saveAs(fn)) log_->logInfo(tr("Saved as: %1").arg(fn));
    }
    void onFnChanged(const QString& fn) {
        if (!fn.isEmpty()) {
            namespace fs = std::filesystem;
            setWindowTitle(QString("%1[*]").arg(QString::fromStdString(fs::path(fn.toStdString()).filename().string())));
        } else setWindowTitle(AppMetadata::NAME);
    }
    void onDirty(bool d) { setWindowModified(d); aSave_->setEnabled(simConf_->isOpen() && d); aClose_->setEnabled(simConf_->isOpen()); aSaveAs_->setEnabled(simConf_->isOpen()); }
    void onLoaded() { setSimConfOpen(true); stack_->syncAllFromConfig(); tree_->setSimConfOpen(true); }
};

#endif // HAS_QT5
} // namespace betsee
#endif // BETSEE_H
