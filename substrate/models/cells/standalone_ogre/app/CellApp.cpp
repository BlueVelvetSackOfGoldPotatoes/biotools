#include "app/CellApp.h"

#include <Bites/OgreTrays.h>
#include <OgreCamera.h>
#include <OgreColourValue.h>
#include <OgreFrameEvent.h>
#include <OgreLight.h>
#include <OgreRoot.h>
#include <OgreSceneManager.h>
#include <OgreSceneNode.h>
#include <OgreViewport.h>
#include <RTShaderSystem/OgreShaderGenerator.h>
#include <SDL_keycode.h>

namespace cells::ogre_view {

CellApp::CellApp()
    : OgreBites::ApplicationContext("CellApp"), sim_(CellInit{}) {
    init_.icosphere_subdivisions = 2;
    init_.fluid_particles = 800;
    init_.params.enable_active_forces = true;
    init_.params.active_force = 0.25;
    init_.params.substrate_friction = 0.85;
    init_.params.substrate_z = -0.75;
    init_.params.dt = 0.005;
    rebuild_sim();
}

void CellApp::rebuild_sim() {
    sim_ = CellSim(init_);
    accumulator_ = 0.0;
}

void CellApp::setup() {
    OgreBites::ApplicationContext::setup();
    addInputListener(this);

    scene_manager_ = getRoot()->createSceneManager();
    bridge_ = std::make_unique<OgreBridge>(scene_manager_);

    auto* shadergen = Ogre::RTShader::ShaderGenerator::getSingletonPtr();
    if (shadergen != nullptr) {
        shadergen->addSceneManager(scene_manager_);
    }

    scene_manager_->setAmbientLight(Ogre::ColourValue(0.4f, 0.4f, 0.45f));
    auto* light = scene_manager_->createLight();
    auto* light_node = scene_manager_->getRootSceneNode()->createChildSceneNode();
    light_node->setPosition(6.0f, 8.0f, 10.0f);
    light_node->attachObject(light);

    camera_ = scene_manager_->createCamera("MainCamera");
    camera_->setNearClipDistance(0.01f);
    camera_->setAutoAspectRatio(true);
    auto* cam_node = scene_manager_->getRootSceneNode()->createChildSceneNode();
    cam_node->setPosition(0.0f, 0.0f, 5.0f);
    cam_node->lookAt(Ogre::Vector3(0.0f, 0.0f, 0.0f), Ogre::Node::TS_WORLD);
    cam_node->attachObject(camera_);

    getRenderWindow()->addViewport(camera_);
    getRoot()->addFrameListener(this);
    bridge_->update(sim_.view());
}

bool CellApp::frameRenderingQueued(const Ogre::FrameEvent& evt) {
    if (paused_) return true;

    accumulator_ += evt.timeSinceLastFrame;
    const double fixed_dt = init_.params.dt;
    while (accumulator_ >= fixed_dt) {
        sim_.step(fixed_dt);
        accumulator_ -= fixed_dt;
    }
    if (bridge_) {
        bridge_->update(sim_.view());
    }
    return true;
}

bool CellApp::keyPressed(const OgreBites::KeyboardEvent& evt) {
    if (evt.keysym.sym == SDLK_SPACE) {
        paused_ = !paused_;
        return true;
    }
    if (evt.keysym.sym == SDLK_r) {
        rebuild_sim();
        return true;
    }
    if (evt.keysym.sym == SDLK_d) {
        sim_.damage_random_vertices(0.12, init_.seed + 111U);
        sim_.damage_random_particles(0.12, init_.seed + 222U);
        return true;
    }
    return false;
}

} // namespace cells::ogre_view
