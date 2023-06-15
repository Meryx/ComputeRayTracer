@group(0) @binding(0) var framebuffer: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var<storage> primitives : array<Sphere>;
@group(0) @binding(2) var<storage, read_write> alt_color_buffer : array<vec3<f32>>;
@group(0) @binding(3) var<uniform> sample : u32;
@group(0) @binding(4) var<storage> planar_patches : array<PlanarPatch>;
@group(0) @binding(5) var<storage> camera : Camera;


const PI: f32 = 3.14159265359;
const roughness: f32 = 0.4;
const metallic: f32 = 0.1;

struct Sphere {
    geometry: vec4<f32>,
    albedo: vec3<f32>,
    index: u32
};

struct Camera {
  origin: vec3<f32>,
  direction: vec3<f32>,
  focal_distance_width_height: vec3<f32>
}

struct PlanarPatch {
  origin: vec3<f32>,
  edge1: vec3<f32>,
  edge2: vec3<f32>,
  albedo: vec3<f32>,
  emission: vec3<f32>,
  index: u32
};

struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>
};

struct HitRecord {
  p: vec3<f32>,
  normal: vec3<f32>,
  t: f32,
  hit: bool,
  index: u32,
  albedo: vec3<f32>,
  emission: vec3<f32>,
  last_index: u32,
};

var<private> light_source : vec3<f32> = vec3<f32>(0.0, 0.0, 0);
var<private> light_color : vec3<f32> = vec3<f32>(1000000.0, 1000000.0, 1000000.0);
var<private> light_center : vec3<f32> = vec3<f32>(278, 554.0, 277);
var<private> light_radius : f32 = 150.0;

@compute 
@workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) GlobalInvocationID: vec3<u32>) {


    let screen_size: vec2<u32> = textureDimensions(framebuffer);
    if(GlobalInvocationID.x >= screen_size.x || GlobalInvocationID.y >= screen_size.y) {
        return;
    }
    let screen_pos : vec2<i32> = vec2<i32>(i32(GlobalInvocationID.x), i32(GlobalInvocationID.y));

    // seed = rnd[GlobalInvocationID.x + GlobalInvocationID.y * screen_size.x];
    seed = vec4<u32>(screen_size.y - u32(screen_pos.y), screen_size.x - u32(screen_pos.x), sample, tea(screen_size.x - u32(screen_pos.x), screen_size.y - u32(screen_pos.y)));
    

    let ambient = vec3<f32>(0.08) * vec3<f32>(1.0);
    var shadow : bool = false;

    let lower_left_corner : vec3<f32> = camera.origin + camera.focal_distance_width_height;
    let horizontal : vec3<f32> = -vec3<f32>(camera.focal_distance_width_height.x * 2, 0, 0);
    let vertical : vec3<f32> = -vec3<f32>(0, camera.focal_distance_width_height.y * 2, 0);
        var c : vec3<f32> = vec3<f32>(0.0);
            var color : vec3<f32> = vec3<f32>(0.0);



    for(var s : u32 = 0; s < 4; s = s + 1)
    {
      let u : f32 = (f32(screen_pos.x) + rand()) / f32(screen_size.x);
      let v : f32 = (f32(screen_size.y) - f32(screen_pos.y) + rand()) / f32(screen_size.y);
      var ray : Ray;
      ray.origin = camera.origin;
      ray.direction = normalize(lower_left_corner + u * horizontal + v * vertical - ray.origin);
      var total : vec3<f32> = vec3<f32>(1);
      var albedo = vec3<f32>(1.0);
      var hit_record : HitRecord;
      hit_record.index = 100;
      hit_record.last_index = 100;

    var attenuation : f32 = 1.0;
    var distance : f32 = 1.0;
    var pref_direction : vec3<f32> = vec3<f32>(0.0);


    var pdf : f32 = 1;
    var radiance : vec3<f32> = vec3<f32>(0.0);
    var throughput : vec3<f32> = vec3<f32>(1.0);

    for(var m : u32 = 0; m < 100; m++)
    {


      hit_record.hit = false;
      hit_record.t = 100000;

      for(var i : u32 = 0; i < arrayLength(&planar_patches); i = i + 1)
      {
        ray_patch_intersection(planar_patches[i], ray, &hit_record);
      }
      // for(var i : u32 = 0; i < arrayLength(&primitives); i = i + 1)
      // {
      //   //ray_sphere_intersection(primitives[i], ray, &hit_record);
      // }
      if(hit_record.hit)
      {


        if(hit_record.index == 2)
        {
          let exponent = 40.0;
          let light_dir = ray.direction;
          let view_dir = normalize(camera.origin - ray.origin);
          let h = normalize(light_dir + view_dir);
          let pref_direction = hit_record.normal;
          let attn = pow((max(dot(normalize(-pref_direction), light_dir), 0.0)), exponent);
          c = hit_record.emission;
          radiance += throughput * hit_record.emission * attn;
          if(m == 0)
          {
            radiance = vec3<f32>(1.0);
          }
          break;
        }

        

        hit_record.last_index = hit_record.index;
        ray.origin = hit_record.p;
        let res = cosine_weighted_sample_hemisphere(hit_record.normal);
        ray.direction = normalize(res.xyz);
        ray.origin = ray.origin + ray.direction * 0.001;
        pdf = res.w;
        albedo = albedo * hit_record.albedo;
        throughput = throughput * albedo;



        
        

      }else{
        c = vec3<f32>(0.0);
        break;
      }
    }


    

    // if(!(attenuation <= 1.0 || attenuation >= 1.0))
    // {
    //   attenuation = 1.0;
    // }
    
    color += radiance;
    }
    //c = c  * (1 / pdf);

    color = color / f32(4);
    c = color;


    var current_color = alt_color_buffer[u32(screen_pos.x) + u32(screen_pos.y) * screen_size.x];
    let combine = current_color + c;
    alt_color_buffer[u32(screen_pos.x) + u32(screen_pos.y) * screen_size.x] = combine;

    c = combine;
    c = c / f32(sample);
    
    c = c / (c + vec3<f32>(1.0));
    c = pow(c, vec3<f32>(1.0 / 2.2));
    
    textureStore(framebuffer, screen_pos, vec4<f32>(c, 1.0));
    
}

fn ray_at(ray : Ray, t : f32) -> vec3<f32>
{
  return ray.origin + t * ray.direction;
}

/* Primitive intersection */
fn ray_sphere_intersection(sphere : Sphere, ray : Ray, hit_record : ptr<function, HitRecord>)
{
  if((*hit_record).last_index == sphere.index)
  {
    return;
  }

  let origin = ray.origin;
  let direction = ray.direction;
  let radius = sphere.geometry.w;
  let center = vec3<f32>(sphere.geometry.xyz);
  let oc = origin - center;
  let a = dot(direction, direction);
  let b = 2.0 * dot(oc, direction);
  let c = dot(oc, oc) - radius * radius;
  let discriminant = b * b - 4.0 * a * c;

  if (discriminant < 0.0) {
    return;
  }

  var t : f32 = (-b - sqrt(discriminant)) / (2.0 * a);
  if (t < 0.0 || t > (*hit_record).t) {
    t = (-b + sqrt(discriminant)) / (2.0 * a);
    if(t < 0.0 || t > (*hit_record).t)
    {
      return;
    }
  }

  (*hit_record).p = ray_at(ray, t);
  (*hit_record).normal = normalize((*hit_record).p - center);
  (*hit_record).t = t;
  (*hit_record).hit = true;
  (*hit_record).index = sphere.index;
  (*hit_record).albedo = sphere.albedo;
}

fn ray_patch_intersection(planar_patch : PlanarPatch, ray : Ray, hit_record : ptr<function, HitRecord>)
{


  if((*hit_record).last_index == planar_patch.index)
  {
    return;
  }

  if((*hit_record).last_index == 1 && planar_patch.index == 2)
  {
    return;
  }

  let origin = ray.origin;
  let direction = ray.direction;
  let edge1 = planar_patch.edge1;
  let edge2 = planar_patch.edge2;
  var normal = normalize(cross(normalize(edge1), normalize(edge2)));
  let origin_to_origin = planar_patch.origin - origin;
  var ndotd = dot(normal, direction);
  if(ndotd > 0)
  {
    normal = -normal;
  }
  ndotd = dot(normal, direction);
  if(abs(ndotd)  < 0.0001)
  {
    return;
  }

  let t = dot(normal, origin_to_origin) / (ndotd);
  if(t < 0.0 || t > (*hit_record).t)
  {
    return;
  }
  let intersection = ray_at(ray, t);
  let m = intersection - planar_patch.origin;
  let u = dot(m, edge1) / dot(edge1, edge1);
  let v = dot(m, edge2) / dot(edge2, edge2);

  if(u < 0.0 || u > 1.0 || v < 0.0 || v > 1.0)
  {
    return;
  }

  (*hit_record).hit = true;
  (*hit_record).p = intersection;
  (*hit_record).normal = normalize(normal);
  (*hit_record).t = t;
  (*hit_record).index = planar_patch.index;
  (*hit_record).albedo = planar_patch.albedo;
  (*hit_record).emission = planar_patch.emission;
}


/* Sampling */

fn uniformally_sample_hemisphere(normal: vec3<f32>) -> vec3<f32> {
   let u = rand();
  let v = rand();
  let theta = 2.0 * PI * u;
  let phi = acos(2.0 * v - 1.0);
  let x = cos(theta) * sin(phi);
  let y = sin(theta) * sin(phi);
  let z = cos(phi);
  var sample = vec3<f32>(x, y, z);
  if(dot(sample, normal) < 0.0)
  {
    sample = -sample;
  }
  return sample;
}

fn cosine_weighted_sample_hemisphere(normal : vec3<f32>) -> vec4<f32> {
  let u = rand();
  let v = rand();
  let r = sqrt(u);
  let theta = 2.0 * PI * v;
  let x = r * cos(theta);
  let y = r * sin(theta);
  let z = sqrt(max(0.0, 1.0 - u));

  var up : vec3<f32>;
  if(abs(normal.z) < 0.999)
  {
    up = vec3<f32>(0.0, 0.0, 1.0);
  }
  else
  {
    up = vec3<f32>(1.0, 0.0, 0.0);
  }
  let tangent = normalize(cross(up, normal));
  let bitangent = cross(normal, tangent);
  let direction = tangent * x + bitangent * y + normal * z;
  return vec4<f32>(direction, z/PI);
}


fn uniformally_sample_rectangle(rect : PlanarPatch) -> vec3<f32>
{
  let u = rand();
  let v = rand();
  let point = rect.origin + u * rect.edge1 + v * rect.edge2;
  return point;
}

fn sample_circle(radius : f32) -> vec2<f32>
{
    let r = sqrt(rand()) * radius;
    let theta = rand() * 2.0 * PI;
    return vec2<f32>(r * cos(theta), r * sin(theta));
}

fn sample_surface_circle(center : vec3<f32>, radius : f32) -> vec3<f32>
{
    let offset = sample_circle(radius);
    return vec3<f32>(center.x + offset.x, center.y , center.z + offset.y);
}


fn sample_circle_cosine_weighted(radius : f32) -> vec2<f32>
{
  let r = sqrt(rand()) * radius;
  let theta = rand() * 2.0 * PI;

  let u = rand();
  let v = rand();
  let phi = 2.0 * PI * u;
  let cosTheta = sqrt(1.0 - v);
  let sinTheta = sqrt(v);

  let x = cos(phi) * sinTheta;
  let y = sin(phi) * sinTheta;
  return vec2<f32>(r * x, r * y);
}


/* Utilties to generate random numbers */

fn tea(val0:u32, val1:u32)->u32{
// "GPU Random Numbers via the Tiny Encryption Algorithm"
  var v0 = val0;
  var v1 = val1;
  var s0 = u32(0);
  for (var n: i32 = 0; n < 16; n++) {
    s0 += 0x9e3779b9;
    v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
    v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
  }

  return v0;
}

fn pcg4d()
{
    seed = seed * 1664525u + 1013904223u;
    seed.x += seed.y * seed.w; 
    seed.y += seed.z * seed.x; 
    seed.z += seed.x * seed.y; 
    seed.w += seed.y * seed.z;
    seed = seed ^ (seed >> vec4<u32>(16u, 16u, 16u, 16u));
    seed.x += seed.y * seed.w; 
    seed.y += seed.z * seed.x; 
    seed.z += seed.x * seed.y; 
    seed.w += seed.y * seed.z;
}

fn rand() -> f32
{
    pcg4d(); 
    return f32(seed.x & 0x00ffffffu) / f32(0x00ffffffu);
}

var<private> seed : vec4<u32> = vec4<u32>(0);