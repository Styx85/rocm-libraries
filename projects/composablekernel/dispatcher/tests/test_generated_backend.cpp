// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/// Unit tests for the supports() gate of the generated GEMM backends.
/// Note: Uses stub kernel structs; no kernel is compiled or launched. The gfx125 gate tests
/// query the current device (no device reads as not gfx125).

#include "ck_tile/ops/gemm/kernel/gemm_kernel.hpp"
#include "ck_tile/dispatcher/backends/generated_kernel_backend.hpp"
#include "ck_tile/dispatcher/backends/generated_tile_backend.hpp"
#include "ck_tile/dispatcher/backends/gfx125_gate.hpp"
#include "ck_tile/dispatcher/dispatcher.hpp"
#include "ck_tile/dispatcher/registry.hpp"
#include "test_mock_kernel.hpp"
#include <gtest/gtest.h>

using namespace ck_tile::dispatcher;
using namespace ck_tile::dispatcher::test;

namespace {

// Fully padded fp16 kernel, as the codegen emits for every fixed-width (_vec) kernel.
template <int A, int B, int C>
struct PaddedStubKernel
{
    using ADataType             = ck_tile::half_t;
    using BDataType             = ck_tile::half_t;
    using CDataType             = ck_tile::half_t;
    using AccDataType           = float;
    static constexpr bool kPadM = true;
    static constexpr bool kPadN = true;
    static constexpr bool kPadK = true;
    static constexpr int TileM  = 128;
    static constexpr int TileN  = 128;
    static constexpr int TileK  = 32;
    struct GemmPipeline
    {
        template <bool = false>
        static constexpr int GetVectorSizeA()
        {
            return A;
        }
        template <bool = false>
        static constexpr int GetVectorSizeB()
        {
            return B;
        }
    };
    static constexpr int VectorSizeC = C;
    static float launch(const ck_tile::GemmHostArgs&, const ck_tile::stream_config&) { return 0.f; }
};

// Same kernel, but its pipeline carries a gfx125-only marker: the TDM flag that
// UniversalGemmKernel::kIsTdmPipeline reads, or a named BarrierPipeline.
struct TdmMarker
{
    static constexpr bool skipCheckValidLaunchParams = true;
};
struct NamedBarrierMarker
{
    struct BarrierPipeline
    {
    };
};
template <typename Marker>
struct Gfx125StubKernel : PaddedStubKernel<8, 8, 8>
{
    struct GemmPipeline : PaddedStubKernel<8, 8, 8>::GemmPipeline, Marker
    {
    };
};

template <int A, int B, int C>
using LegacyInstance = backends::GeneratedKernelInstance<PaddedStubKernel<A, B, C>>;
template <int A, int B, int C>
using TileInstance = backends::GeneratedTileKernelInstance<PaddedStubKernel<A, B, C>,
                                                           ck_tile::half_t,
                                                           ck_tile::half_t,
                                                           ck_tile::half_t,
                                                           float>;

KernelKey make_vec_key()
{
    KernelKey key               = make_test_key(128, 128, 32, "gfx950");
    key.algorithm.vector_size_a = 1;
    key.algorithm.vector_size_b = 1;
    key.algorithm.vector_size_c = 8;
    return key;
}

// Padding must not hide the global vector widths: native fp16 rcr needs K % 8,
// the _vec1_1_8 kernel only needs N % 8.
template <template <int, int, int> class Instance>
void expect_vector_gate()
{
    const Instance<8, 8, 8> native(make_test_key(128, 128, 32, "gfx950"), "native");
    const Instance<1, 1, 8> vec(make_vec_key(), "vec");

    EXPECT_TRUE(native.supports(Problem(512, 512, 512)));
    EXPECT_FALSE(native.supports(Problem(512, 512, 257)));
    EXPECT_TRUE(vec.supports(Problem(512, 512, 257)));
    EXPECT_FALSE(vec.supports(Problem(512, 129, 257)));

    // Native keys keep zero widths and no suffix, even when the pipeline's
    // tile distribution loads fewer than 16 bytes or the epilogue stores scalar C.
    const auto key = make_test_key(128, 128, 32, "gfx950");
    const Instance<4, 4, 8> small_tile(key, "native_small_tile");
    EXPECT_TRUE(small_tile.supports(Problem(256, 256, 516)));
    EXPECT_FALSE(small_tile.supports(Problem(256, 256, 514)));
    EXPECT_EQ(small_tile.get_key().encode_identifier().find("_vec"), std::string::npos);

    const Instance<8, 8, 1> default_epilogue(key, "native_default");
    EXPECT_TRUE(default_epilogue.supports(Problem(256, 257, 512)));
    EXPECT_FALSE(default_epilogue.supports(Problem(256, 257, 516)));
}

} // anonymous namespace

TEST(GeneratedBackendTest, GeneratedKernelInstanceGatesVectorWidths)
{
    expect_vector_gate<LegacyInstance>();
}

TEST(GeneratedBackendTest, GeneratedTileKernelInstanceGatesVectorWidths)
{
    expect_vector_gate<TileInstance>();
}

// The register_all_kernels.hpp wrappers use GeneratedKernelInstance; first-fit
// must skip the native kernel and pick the narrower one, or return none.
TEST(GeneratedBackendTest, FirstFitFallsThroughToNarrowerKernel)
{
    Registry registry;
    auto native =
        std::make_shared<LegacyInstance<8, 8, 8>>(make_test_key(128, 128, 32, "gfx950"), "native");
    auto vec = std::make_shared<LegacyInstance<1, 1, 8>>(make_vec_key(), "vec");
    ASSERT_TRUE(registry.register_kernel(native));
    ASSERT_TRUE(registry.register_kernel(vec));

    Dispatcher dispatcher(&registry);
    EXPECT_EQ(dispatcher.select_kernel(Problem(512, 512, 257)), vec);
    EXPECT_EQ(dispatcher.select_kernel(Problem(512, 129, 257)), nullptr);
}

static_assert(!backends::pipeline_requires_gfx125<PaddedStubKernel<8, 8, 8>::GemmPipeline>);
static_assert(backends::pipeline_requires_gfx125<Gfx125StubKernel<TdmMarker>::GemmPipeline>);
static_assert(
    backends::pipeline_requires_gfx125<Gfx125StubKernel<NamedBarrierMarker>::GemmPipeline>);

// A gfx125-only kernel is selectable exactly when the current device is gfx125 (the same
// ck_tile::is_gfx125_supported() check IsSupportedArgument makes), so a mixed-target registry
// falls through to a kernel valid for this GPU instead of throwing at launch. Holds with or
// without a GPU: no device reads as not gfx125.
template <typename Marker>
void expect_gfx125_gate()
{
    using Tile   = backends::GeneratedTileKernelInstance<Gfx125StubKernel<Marker>,
                                                         ck_tile::half_t,
                                                         ck_tile::half_t,
                                                         ck_tile::half_t,
                                                         float>;
    using Legacy = backends::GeneratedKernelInstance<Gfx125StubKernel<Marker>>;

    const bool on_gfx125 = ck_tile::is_gfx125_supported();
    const auto key       = make_test_key(128, 128, 32, "gfx1250");
    const Problem problem(512, 512, 512);
    EXPECT_EQ(Tile(key, "tile").supports(problem), on_gfx125);
    EXPECT_EQ(Legacy(key, "legacy").supports(problem), on_gfx125);

    Registry registry;
    auto gated = std::make_shared<Legacy>(key, "gated");
    auto plain =
        std::make_shared<LegacyInstance<8, 8, 8>>(make_test_key(128, 128, 64, "gfx950"), "plain");
    ASSERT_TRUE(registry.register_kernel(gated));
    ASSERT_TRUE(registry.register_kernel(plain));

    Dispatcher dispatcher(&registry);
    const auto selected = dispatcher.select_kernel(problem);
    if(on_gfx125)
        EXPECT_TRUE(selected == gated || selected == plain);
    else
        EXPECT_EQ(selected, plain);
}

TEST(GeneratedBackendTest, TdmPipelineGatedOnGfx125Device) { expect_gfx125_gate<TdmMarker>(); }

TEST(GeneratedBackendTest, NamedBarrierPipelineGatedOnGfx125Device)
{
    expect_gfx125_gate<NamedBarrierMarker>();
}

// A Tile Engine SelectedKernel declares its GemmPipeline inside launch(), so it publishes
// kRequiresGfx125 at struct scope for TileKernelInstance::supports(). A kernel with neither that
// flag nor a GemmPipeline is left to its own IsSupportedArgument.
template <bool Requires>
struct TileEngineStubKernel
{
    static constexpr bool kRequiresGfx125 = Requires;
};
struct OpaqueStubKernel
{
};

TEST(GeneratedBackendTest, TileEngineKernelGatedOnGfx125Device)
{
    EXPECT_EQ(backends::device_supports_kernel<TileEngineStubKernel<true>>(),
              ck_tile::is_gfx125_supported());
    EXPECT_TRUE(backends::device_supports_kernel<TileEngineStubKernel<false>>());
    EXPECT_TRUE(backends::device_supports_kernel<OpaqueStubKernel>());
    EXPECT_EQ(backends::device_supports_kernel<Gfx125StubKernel<TdmMarker>>(),
              ck_tile::is_gfx125_supported());
}
