#include "render/OgreBridge.h"

namespace cells::ogre_view {

OgreBridge::OgreBridge(Ogre::SceneManager* scene_manager)
    : membrane_(scene_manager), particles_(scene_manager) {}

void OgreBridge::update(const SimView& view) {
    membrane_.update(view);
    particles_.update(view);
}

} // namespace cells::ogre_view
