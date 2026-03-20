#pragma once

#include "models/cells/src/sim/sim_state.h"

namespace Ogre {
class ManualObject;
class SceneManager;
class SceneNode;
}

namespace cells::ogre_view {

class ParticlesRenderable {
public:
    explicit ParticlesRenderable(Ogre::SceneManager* scene_manager);
    void update(const SimView& view);

private:
    Ogre::ManualObject* object_ = nullptr;
    Ogre::SceneNode* node_ = nullptr;
};

} // namespace cells::ogre_view
