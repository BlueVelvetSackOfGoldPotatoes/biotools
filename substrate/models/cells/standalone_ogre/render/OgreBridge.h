#pragma once

#include "models/cells/src/sim/sim_state.h"
#include "render/MembraneRenderable.h"
#include "render/ParticlesRenderable.h"

namespace Ogre {
class SceneManager;
}

namespace cells::ogre_view {

class OgreBridge {
public:
    explicit OgreBridge(Ogre::SceneManager* scene_manager);
    void update(const SimView& view);

private:
    MembraneRenderable membrane_;
    ParticlesRenderable particles_;
};

} // namespace cells::ogre_view
