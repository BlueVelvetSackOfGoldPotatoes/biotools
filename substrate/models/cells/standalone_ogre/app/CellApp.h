#pragma once

#include "models/cells/src/cell_models.h"
#include "render/OgreBridge.h"

#include <memory>

#include <Bites/OgreApplicationContext.h>
#include <Bites/OgreInput.h>
#include <OgreFrameListener.h>

namespace Ogre {
class Camera;
class SceneManager;
}

namespace cells::ogre_view {

class CellApp final : public OgreBites::ApplicationContext,
                      public OgreBites::InputListener,
                      public Ogre::FrameListener {
public:
    CellApp();

    void setup() override;
    bool frameRenderingQueued(const Ogre::FrameEvent& evt) override;
    bool keyPressed(const OgreBites::KeyboardEvent& evt) override;

private:
    void rebuild_sim();

    CellInit init_;
    CellSim sim_;
    std::unique_ptr<OgreBridge> bridge_;
    Ogre::SceneManager* scene_manager_ = nullptr;
    Ogre::Camera* camera_ = nullptr;
    bool paused_ = false;
    double accumulator_ = 0.0;
};

} // namespace cells::ogre_view
