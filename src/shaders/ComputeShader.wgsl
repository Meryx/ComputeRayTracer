@group(0) @binding(0) var framebuffer: texture_storage_2d<rgba8unorm, write>;

@compute 
@workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) GlobalInvocationID: vec3<u32>) {

    

    let screenSize: vec2<u32> = textureDimensions(framebuffer);
    let screenPos : vec2<i32> = vec2<i32>(i32(GlobalInvocationID.x), i32(GlobalInvocationID.y));

    seed = vec4<u32>(u32(screenPos.x), u32(screenPos.y), 2u, u32(screenPos.x) + u32(screenPos.y));

    //seed = tea(u32(screenPos.x), u32(screenPos.y * 800));
    //let res = pcg3d(vec3<u32>(u32(screenPos.x), u32(screenPos.y * 1000), seed));
    //let c = vec3<f32>(f32(res.x) / f32(0x00010000), f32(res.y) / f32(0x00010000), f32(res.z) / f32(0x00010000));
    let c = vec3<f32>(rand(), rand(), rand());

    textureStore(framebuffer, screenPos, vec4<f32>(c, 1.0));
    
}


fn pcg4d()
{
    seed = seed * 1664525u + 1013904223u;
    seed.x += seed.y * seed.w; seed.y += seed.z * seed.x; seed.z += seed.x * seed.y; seed.w += seed.y * seed.z;
    seed = seed ^ (seed >> vec4<u32>(16u, 16u, 16u, 16u));
    seed.x += seed.y * seed.w; 
    seed.y += seed.z * seed.x; 
    seed.z += seed.x * seed.y; 
    seed.w += seed.y * seed.z;
}

fn rand() -> f32
{
    pcg4d(); 
    return f32(seed.x) / f32(0xffffffffu);
}

var<private> seed : vec4<u32> = vec4<u32>(0);