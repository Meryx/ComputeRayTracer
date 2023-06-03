/* File: TextureRenderShader.wgsl
   This shader renders a texture onto a screen-filled quad. */

// Samplers and Bindings
@group(0) @binding(0) var screen_sampler: sampler;
@group(0) @binding(1) var framebuffer: texture_2d<f32>;

// Vertex Output Structure
struct VertexOutput {
    @builtin(position) Position: vec4<f32>,  // Position of the vertex
    @location(0) TexCoord: vec2<f32>,  // Texture coordinates of the vertex
}

// Vertex Shader
@vertex
fn vert_main (@builtin(vertex_index) index: u32) -> VertexOutput {
    /* Vertex positions of the screen-filled quad */
    let positions = array<vec2<f32>, 6>(
        vec2<f32>(1.0,  1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0,  1.0),
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0,  -1.0)
    );

    /* Texture coordinates for each vertex of the quad */
    let texCoords = array<vec2<f32>, 6>(
        vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 0.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 1.0)
    );

    var output: VertexOutput;
    output.Position = vec4<f32>(positions[index], 0.0, 1.0);
    output.TexCoord = texCoords[index];
    return output;
}

// Fragment Shader
@fragment
fn frag_main (input: VertexOutput) -> @location(0) vec4<f32> {
    /* Sample the texture using the screen_sampler and input texture coordinates */
    return textureSample(framebuffer, screen_sampler, input.TexCoord);
}
