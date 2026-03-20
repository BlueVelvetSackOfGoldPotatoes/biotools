#include "render/ParticlesRenderable.h"

#include <OgreManualObject.h>
#include <OgreRenderOperation.h>
#include <OgreSceneManager.h>
#include <OgreSceneNode.h>

namespace cells::ogre_view {

ParticlesRenderable::ParticlesRenderable(Ogre::SceneManager* scene_manager) {
    object_ = scene_manager->createManualObject();
    object_->setDynamic(true);
    node_ = scene_manager->getRootSceneNode()->createChildSceneNode();
    node_->attachObject(object_);
}

void ParticlesRenderable::update(const SimView& view) {
    if (object_ == nullptr) return;
    object_->clear();
    object_->begin("Cells/Particles", Ogre::RenderOperation::OT_POINT_LIST);
    for (std::size_t i = 0; i < view.particle_positions.size(); ++i) {
        const auto& p = view.particle_positions[i];
        object_->position(static_cast<float>(p.x), static_cast<float>(p.y), static_cast<float>(p.z));
    }
    object_->end();
}

} // namespace cells::ogre_view
