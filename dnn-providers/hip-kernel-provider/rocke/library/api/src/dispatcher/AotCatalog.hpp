// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

#include "dispatcher/AotInstance.hpp"

namespace rocke_client::dispatcher
{

// The set of AOT-built kernel instances the dispatcher can select from.
//
// PHASE 1 (this ticket): the production catalog is ALWAYS EMPTY. The rocKE AOT
// producer installs loose per-kernel .co + .sidecar.json files under
// <plugin_dir>/arch_content/rocke/<arch>/. The remaining TODO is the runtime
// loader that reads those files and populates the catalog. Until that lands,
// `loadDefault()` returns an empty catalog and the engine therefore declines
// every graph (a deliberate no-op). The selection logic is still real and fully
// exercised in unit tests via catalogs constructed from fixture instances.
class AotCatalog
{
public:
    AotCatalog() = default;
    explicit AotCatalog(std::vector<AotInstance> instances);

    // The production catalog source.
    //
    // TODO(kpack): implement the runtime loader. This must:
    //   1. resolve the loaded plugin's directory and the per-arch artifact root
    //      <plugin_dir>/arch_content/rocke/<arch>/ (see defaultArtifactRoot());
    //   2. enumerate that arch's *.co + *.sidecar.json pairs (the installed
    //      source of truth; aot_list.json is a build-time input, not installed);
    //   3. parse each sidecar instance (compile_spec + selection +
    //      attribute_constraints) into AotInstance, mirroring
    //      rocke_client_aot.instance_schema semantics.
    // For now it logs the deferral and returns an empty catalog.
    static AotCatalog loadDefault();

    // Instances whose op and arch match, in stable (insertion) order. Returns
    // non-owning references borrowed from this catalog (valid for its lifetime,
    // which spans the owning engine) so the selection path copies no instances.
    std::vector<std::reference_wrapper<const AotInstance>>
        candidatesFor(const std::string& op, const std::string& arch) const;

    bool empty() const
    {
        return _instances.empty();
    }

    std::size_t size() const
    {
        return _instances.size();
    }

private:
    std::vector<AotInstance> _instances;
};

// The plugin-relative root under which installed per-arch rocKE AOT bundles live:
// <plugin_dir>/arch_content/rocke/<arch>/, where arch_content is a generic
// per-arch content container (other engines get sibling subdirs). loadDefault()
// resolves the loaded plugin's directory at runtime and appends the device arch.
// This is a path constant only; nothing reads from it yet (Phase-1 catalog empty).
const char* defaultArtifactRoot();

} // namespace rocke_client::dispatcher
