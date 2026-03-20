#include "app/CellApp.h"

int main() {
    cells::ogre_view::CellApp app;
    app.initApp();
    app.getRoot()->startRendering();
    app.closeApp();
    return 0;
}
