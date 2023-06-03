@group(0) @binding(0) var framebuffer: texture_storage_2d<rgba8unorm, write>;

@compute 
@workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) GlobalInvocationID: vec3<u32>) {
    let screenSize: vec2<u32> = textureDimensions(framebuffer);
    let screenPos : vec2<i32> = vec2<i32>(i32(GlobalInvocationID.x), i32(GlobalInvocationID.y));

    if(screenPos.x < 800 || screenPos.x > 1600 || screenPos.y < 400 || screenPos.y > 600) {
        textureStore(framebuffer, screenPos, vec4<f32>(1.0, 0.0, 0.0, 1.0));
    }else {
        textureStore(framebuffer, screenPos, vec4<f32>(0.0, 1.0, 1.0, 1.0));
    }
    
}