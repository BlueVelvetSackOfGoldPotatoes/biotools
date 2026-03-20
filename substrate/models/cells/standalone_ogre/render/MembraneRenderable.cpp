#include "render/MembraneRenderable.h"

#include <OgreManualObject.h>
#include <OgreRenderOperation.h>
#include <OgreSceneManager.h>
#include <OgreSceneNode.h>

namespace cells::ogre_view {

MembraneRenderable::MembraneRenderable(Ogre::SceneManager* scene_manager) {
    object_ = scene_manager->createManualObject();
    object_->setDynamic(true);
    node_ = scene_manager->getRootSceneNode()->createChildSceneNode();
    node_->attachObject(object_);
}

void MembraneRenderable::update(const SimView& view) {
    if (object_ == nullptr) return;
    object_->clear();
    object_->begin("Cells/Membrane", Ogre::RenderOperation::OT_TRIANGLE_LIST);
    for (std::size_t i = 0; i < view.membrane_positions.size(); ++i) {
        const auto& p = view.membrane_positions[i];
        const auto& n = view.membrane_normals[i];
        object_->position(static_cast<float>(p.x), static_cast<float>(p.y), static_cast<float>(p.z));
        object_->normal(static_cast<float>(n.x), static_cast<float>(n.y), static_cast<float>(n.z));
    }
    for (std::size_t i = 0; i < view.membrane_indices.size(); ++i) {
        object_->index(view.membrane_indices[i]);
    }
    object_->end();
}

} // namespace cells::ogre_view
