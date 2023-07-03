@group(0) @binding(0) var framebuffer: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var<storage> spheres : array<Sphere>;
@group(0) @binding(2) var<storage, read_write> alt_color_buffer : array<vec3<f32>>;
@group(0) @binding(3) var<uniform> sample : u32;
@group(0) @binding(4) var<storage> planar_patches : array<PlanarPatch>;
@group(0) @binding(5) var<storage> camera : Camera;
@group(0) @binding(6) var<storage> CIE : array<array<f32,471>,3>;
@group(0) @binding(7) var<storage> spectra : array<array<f32, 301>>;

const PI: f32 = 3.14159265359;
const MAXDEPTH : u32 = 1000000000;
const INFINITY : f32 = 3.40282346638528859812e+38f;
const GRID_SIZE : u32 = 16;
const BRDF : f32 = 1.0 / PI;
const lambda_min : f32 = 400.0;
const lambda_max : f32 = 700.0;

const DIFFUSE : u32 = 0;
const LIGHT : u32 = 1;

    
struct Sphere {
  center: vec3<f32>,
  radius: f32,
  albedo: vec3<f32>,
  index: u32,
  emission: vec3<f32>
};

struct Camera {
  eye: vec3<f32>,
  lookat: vec3<f32>,
  up: vec3<f32>,
  viewport_width: f32,
  viewport_height: f32,
  focal_length: f32
};

struct PlanarPatch {
  origin: vec3<f32>,
  edge1: vec3<f32>,
  edge2: vec3<f32>,
  emission_index: u32,
  reflectance_index: u32,
  material: u32,
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

struct ShapeIntersection {
  position: vec3<f32>,
  normal: vec3<f32>,
  t: f32,
  hit: bool,
  index: u32,
  albedo: vec3<f32>,
  emission_index: u32,
  last_index: u32,
  reflectance_index: u32,
  material: u32
};

fn camera_basis() -> mat3x3<f32> {
  let w : vec3<f32> = normalize(camera.eye - camera.lookat);
  let u : vec3<f32> = normalize(cross(camera.up, w));
  let v : vec3<f32> = cross(w, u);
  return mat3x3<f32>(u, v, w);
}

fn camera_ray(screen_pos : vec2<u32>, screen_size : vec2<u32>) -> Ray {
  let aspect_ratio : f32 = camera.viewport_width / camera.viewport_height;
  let viewport_height : f32 = 2.0 * tan(camera.focal_length / 2.0);
  let viewport_width : f32 = aspect_ratio * viewport_height;
  let basis = camera_basis();
  let u : vec3<f32> = basis[0];
  let v : vec3<f32> = basis[1];
  let w : vec3<f32> = basis[2];
  let horizontal : vec3<f32> = viewport_width * u;
  let vertical : vec3<f32> = viewport_height * v;
  let lower_left_corner : vec3<f32> = camera.eye - horizontal / 2.0 - vertical / 2.0 - w;

  let camera_film_parameters : vec2<f32> = camera_film_parameters(screen_pos, screen_size);
  let s = camera_film_parameters.x;
  let t = camera_film_parameters.y;

  return Ray(camera.eye, normalize(lower_left_corner + s * horizontal + t * vertical - camera.eye));
}

fn camera_film_parameters(screen_pos : vec2<u32>, screen_size : vec2<u32>) -> vec2<f32> {
  let s : f32 = (f32(screen_pos.x) + (f32(sample % GRID_SIZE) + rand()) / f32(GRID_SIZE)) / f32(screen_size.x);
  let t : f32 = (f32(screen_size.y) - f32(screen_pos.y) + (f32(sample % GRID_SIZE) + rand()) / f32(GRID_SIZE)) / f32(screen_size.y);
  return vec2<f32>(s,t);
}

fn sample_spectrum(spectrum : array<f32, 301>,  lambdas : vec4<u32>) -> vec4<f32>
{
  return vec4<f32>(spectrum[lambdas.x], spectrum[lambdas.y], spectrum[lambdas.z], spectrum[lambdas.w]);
}

fn sample_wavelengths() -> vec4<u32>
{
  let u = rand();
  let range : u32 = u32(lambda_max - lambda_min);
  let lambda : u32 = u32(mix(0, lambda_max - lambda_min, u));

  return vec4<u32>(lambda, (lambda + 4) % range, (lambda + 8) % range, (lambda + 12) % range);
}

fn path_trace(input_ray : Ray) -> vec3<f32>
{
  var ray : Ray = input_ray;
  var depth : u32 = 0;
  var radiance : vec4<f32> = vec4<f32>(0.0);
  var beta : vec4<f32> = vec4<f32>(1.0);
  var pdf_b : f32 = 1.0;
  var exclude : u32 = 100;
  var prev_intersection : ShapeIntersection;
  let lambdas = sample_wavelengths();
  let spec = sample_spectrum(spectra[3], lambdas);

  while(true)
  {
    let shape_intersection : ShapeIntersection = intersect(ray, exclude);
    if(!shape_intersection.hit)
    {
      break;
    }
    exclude = shape_intersection.index;


    var material = shape_intersection.material;
    if(material == LIGHT)
    {
      var le2 = sample_spectrum(spectra[shape_intersection.emission_index], lambdas);
      if(depth == 0)
      {
        radiance += beta * le2;
      } else {
        let light = planar_patches[shape_intersection.index];
        let area = length(light.edge1) * length(light.edge2);
        let abs_cos_theta_l = abs(dot(shape_intersection.normal, -ray.direction));
        let distance = length(prev_intersection.position - shape_intersection.position);
        let distance_squared = pow(distance, 2);
        let pdf_l_area = 1.0 / area;
        let geometric_term = abs_cos_theta_l / distance_squared;
        let pdf_l = pdf_l_area / geometric_term;
        let weight_b = power_heuristic(1, pdf_b, 1, pdf_l);
        radiance += weight_b * le2 * beta;
      }
      break;
    }

    if(depth >= MAXDEPTH)
    {
      break;
    }

    

    beta *= sample_spectrum(spectra[shape_intersection.reflectance_index], lambdas);

    let light = planar_patches[2];
    var normal = normalize(cross((light.edge1), (light.edge2)));
    let ndotd = dot(normal, ray.direction);
    if(ndotd > 0)
    {
      normal = -normal;
    }
    let light_normal = normal;
    let area = length(light.edge1) * length(light.edge2);
    let point_on_light = sample_light();
    let is_visible = is_visible(shape_intersection.position, point_on_light, exclude);
    let light_dir = normalize(point_on_light - shape_intersection.position);
    let cos_theta_i = max(0, dot(shape_intersection.normal, light_dir));

    let radiance_light = spec * cos_theta_i;

    let abs_cos_theta_l = max(0.000001, abs(dot(light_normal, -light_dir)));
    let distance = length(point_on_light - shape_intersection.position);
    let distance_squared = pow(distance, 2);
    let pdf_l_area = 1.0 / area;
    let geometric_term = abs_cos_theta_l / distance_squared;
    
    let pdf_l = pdf_l_area / geometric_term;

    pdf_b = cos_theta_i / PI;
    let weight_l = power_heuristic(1, pdf_l, 1, pdf_b);
    
    radiance += BRDF * (is_visible * radiance_light) * weight_l * beta / pdf_l;

    let new_direction_with_pdf = cosine_weighted_sample_hemisphere(shape_intersection.normal);
    let new_direction = new_direction_with_pdf.xyz;
    pdf_b = new_direction_with_pdf.w;

    let cosnew = abs(dot(shape_intersection.normal, new_direction));

    beta *= BRDF * cosnew / (pdf_b);

    let max_beta_component = max(beta.x, max(beta.y, beta.z));
    if(depth > 1 && max_beta_component < 1)
    {
      let q = max(0, 1 - max_beta_component);
      if(rand() < q)
      {
        break;
      }
      beta /= 1 - q;
    }

    prev_intersection = shape_intersection;

    ray.origin = shape_intersection.position;
    ray.direction = new_direction;
    depth++;
  }
  var r = spectral_to_xyz(radiance, lambdas);
  //r =  white_balance(r, spec, lambdas);
  return r;
}

@compute 
@workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) GlobalInvocationID: vec3<u32>) {


    /* Intialize screen size and position. Return if outside bounds of framebuffer. */
    let screen_size: vec2<u32> = vec2<u32>(u32(camera.viewport_width), u32(camera.viewport_height));
    let screen_pos : vec2<u32> = vec2<u32>(u32(GlobalInvocationID.x), u32(GlobalInvocationID.y));
    if(GlobalInvocationID.x >= screen_size.x || GlobalInvocationID.y >= screen_size.y) {
        return;
    }

    /* Initialize random seed. Multiply x by 100 and y by 100 to ensure unique values between every pixel */
    seed = vec4<u32>(screen_pos.y, screen_pos.x * 100, sample, tea(screen_pos.x, screen_pos.y * 100));



    var ray : Ray = camera_ray(screen_pos, screen_size);
    var c : vec3<f32> = path_trace(ray);
    
    let pixel_index = screen_pos.x + screen_pos.y * screen_size.x;

    var xyzc = c;


    alt_color_buffer[pixel_index] += xyzc;
    xyzc = alt_color_buffer[pixel_index];
    xyzc = xyzc / f32(sample);

    xyzc = xyz_to_rgb(xyzc);


   

    //xyzc = xyzc / (xyzc + vec3<f32>(1.0));
    //xyzc = pow(xyzc, vec3<f32>(1.0 / 2.2));

    textureStore(framebuffer, screen_pos, vec4<f32>(xyzc, 1.0));
    
}

fn white_balance(xyz : vec3<f32> , source_spec : vec4<f32>, lambdas : vec4<u32>) -> vec3<f32>
{
  // let M_bradford : mat3x3<f32> = mat3x3<f32>(0.8951,  0.2664, -0.1614,
  //       -0.7502,  1.7135,  0.0367,
  //        0.0389, -0.0685,  1.0296);

  // let M_bradford_inv : mat3x3<f32> = mat3x3<f32>(0.9869929, -0.1470543,  0.1599627,
  //       0.4323053,  0.5183603,  0.0492912,
  //      -0.0085287,  0.0400428,  0.9684867);

  // let source_illuminant_xyz = spectral_to_xyz_m();
  // //let source_illuminant_xyz = vec3<f32>(0.7, 1.1, 0.5);
  // let target_illuminant_xyz = vec3<f32>(0.95047, 1.0, 1.08883);
  
  // let source_cone = M_bradford * source_illuminant_xyz;
  // let target_cone = M_bradford * target_illuminant_xyz;

  // var diag : mat3x3<f32> = mat3x3<f32>(target_cone.x / source_cone.x, 0.0, 0.0,
  //                                       0.0, target_cone.y / source_cone.y, 0.0,
  //                                       0.0, 0.0, target_cone.z / source_cone.z);

  // let color_cone = M_bradford_inv * diag * M_bradford * xyz;

  // return color_cone;
  return vec3<f32>(1.0);
}

fn is_visible(p0 : vec3<f32>, p1 : vec3<f32>, exclude : u32) -> f32
{
  var ray : Ray;
  ray.origin = p0;
  ray.direction = normalize(p1 - p0);
  let shape_intersection : ShapeIntersection = intersect(ray, exclude);
  if(shape_intersection.hit)
  {
    if(length(shape_intersection.position - p1) < 0.0001)
    {
      return 1.0;
    }
  }
  return 0.0;
}

fn sample_light() -> vec3<f32>
{
  let light = planar_patches[2];
  let u = rand();
  let v = rand();
  let p = light.origin + u * light.edge1 + v * light.edge2;
  return p;
}

fn power_heuristic(nf : f32, f_pdf : f32, ng : f32, g_pdf : f32) -> f32
{
  let f = nf * f_pdf;
  let g = ng * g_pdf;
  return (f * f) / (f * f + g * g);
}

fn intersect(ray : Ray, exclude : u32) -> ShapeIntersection
{
  var shape_intersection : ShapeIntersection;
  shape_intersection.hit = false;

  var t_max : f32 = INFINITY;
  for(var i : u32; i < arrayLength(&planar_patches); i++)
  {
    let temp_si = ray_patch_intersection_test(planar_patches[i], ray, 0.001, t_max, exclude);
    if(temp_si.hit)
    {
      shape_intersection = temp_si;
      t_max = temp_si.t;
    }
    
  }
  for(var i : u32; i < arrayLength(&spheres); i++)
  {
    let temp_si = ray_sphere_intersection(spheres[i], ray, 0.001, t_max, exclude);
    if(temp_si.hit)
    {
      shape_intersection = temp_si;
      t_max = temp_si.t;
    }
    
  }
  return shape_intersection;
}

fn ray_at(ray : Ray, t : f32) -> vec3<f32>
{
  return ray.origin + t * ray.direction;
}

fn ray_sphere_intersection(sphere : Sphere, ray : Ray, t_min : f32, t_max : f32, exclude : u32) -> ShapeIntersection
{
  var shape_intersection : ShapeIntersection;
  shape_intersection.hit = false;

  if(exclude == sphere.index)
  {
    return shape_intersection;
  }

  let origin = ray.origin;
  let direction = ray.direction;
  let radius = sphere.radius;
  let center = sphere.center;
  let oc = origin - center;
  let a = dot(direction, direction);
  let b = 2.0 * dot(oc, direction);
  let c = dot(oc, oc) - radius * radius;
  let discriminant = b * b - 4.0 * a * c;

  if (discriminant < 0.0) {
    return shape_intersection;
  }

  var t : f32 = (-b - sqrt(discriminant)) / (2.0 * a);
  if (t < t_min || t > t_max) {
    t = (-b + sqrt(discriminant)) / (2.0 * a);
    if (t < t_min || t > t_max)
    {
      return shape_intersection;
    }
  }

  shape_intersection.position = ray_at(ray, t);
  shape_intersection.normal = normalize(shape_intersection.position - center);
  shape_intersection.t = t;
  shape_intersection.hit = true;
  shape_intersection.index = sphere.index;
  shape_intersection.albedo = sphere.albedo;
  //shape_intersection.emission = sphere.emission;
  return shape_intersection;
}

fn ray_patch_intersection_test(planar_patch : PlanarPatch, ray : Ray, t_min : f32, t_max : f32, exclude : u32) -> ShapeIntersection
{

  var shape_intersection : ShapeIntersection;
  shape_intersection.hit = false;

  if(exclude == planar_patch.index)
  {
    return shape_intersection;
  }


  let origin = ray.origin;
  let direction = ray.direction;
  let edge1 = planar_patch.edge1;
  let edge2 = planar_patch.edge2;
  var normal = normalize(cross((edge1), (edge2)));
  let origin_to_origin = planar_patch.origin - origin;
  var ndotd = dot(normal, direction);
  if(ndotd > 0)
  {
    normal = -normal;
  }
  ndotd = dot(normal, direction);
  if(abs(ndotd)  < 0.0001)
  {
    return shape_intersection;
  }

  let t = dot(normal, origin_to_origin) / (ndotd);
  if(t < t_min || t > t_max)
  {
    return shape_intersection;
  }
  let intersection = ray_at(ray, t);
  let m = intersection - planar_patch.origin;
  let u = dot(m, edge1) / dot(edge1, edge1);
  let v = dot(m, edge2) / dot(edge2, edge2);

  if(u < 0.0 || u > 1.0 || v < 0.0 || v > 1.0)
  {
    return shape_intersection;
  }

  shape_intersection.hit = true;
  shape_intersection.position = intersection;
  shape_intersection.normal = normalize(normal);
  shape_intersection.t = t;
  shape_intersection.index = planar_patch.index;
  shape_intersection.emission_index = planar_patch.emission_index;
  shape_intersection.reflectance_index = planar_patch.reflectance_index;
  shape_intersection.material = planar_patch.material;
  return shape_intersection;
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
    return f32(seed.x & 0x00ffffffu) / f32(0x01000000);
}

fn sample_CIE_X(lambdas : vec4<u32>) -> vec4<f32>
{
  let CIE_X = CIE[0];
  return vec4<f32>(CIE_X[lambdas.x + 40], CIE_X[lambdas.y + 40], CIE_X[lambdas.z + 40], CIE_X[lambdas.w + 40]);
}

fn sample_CIE_Y(lambdas : vec4<u32>) -> vec4<f32>
{
  let CIE_Y = CIE[1];
  return vec4<f32>(CIE_Y[lambdas.x + 40], CIE_Y[lambdas.y + 40], CIE_Y[lambdas.z + 40], CIE_Y[lambdas.w + 40]);
}

fn sample_CIE_Z(lambdas : vec4<u32>) -> vec4<f32>
{
  let CIE_Z = CIE[2];
  return vec4<f32>(CIE_Z[lambdas.x + 40], CIE_Z[lambdas.y + 40], CIE_Z[lambdas.z + 40], CIE_Z[lambdas.w + 40]);
}


fn spectral_to_xyz(radianc : vec4<f32>, lambdas : vec4<u32>) -> vec3<f32>
{
    var xyz = vec3<f32>(0.0, 0.0, 0.0);
    let X_BAR = sample_CIE_X(lambdas);
    let Y_BAR = sample_CIE_Y(lambdas);
    let Z_BAR = sample_CIE_Z(lambdas);
    let integ = 106.856895;

    let radiance = radianc * 300;

    xyz.x = (dot(X_BAR, radiance) /4)/integ;
    xyz.y = (dot(Y_BAR, radiance) /4)/integ;
    xyz.z = (dot(Z_BAR, radiance) /4)/integ;

    return xyz;
}

fn xyz_to_rgb(rgb : vec3<f32>) -> vec3<f32>
{
    //var rgb = rgbb * 3.2;
    var r =  3.2404542 * rgb.x + -1.5371385 * rgb.y + -0.4985314 * rgb.z;
    var g = -0.9692660 * rgb.x +  1.8760108 * rgb.y +  0.0415560 * rgb.z;
    var b =  0.0556434 * rgb.x + -0.2040259 * rgb.y +  1.0572252 * rgb.z;

    var rr = vec3<f32>(r, g, b);
    rr = vec3(1.0) - exp(-rr * 3.2);


    if(r < 0.0031308)
    {
        rr.r = 12.92 * rr.r;
    }
    else
    {
        rr.r = 1.055 * pow(rr.r, 1.0/2.4) - 0.055;
    }

    if(g < 0.0031308)
    {
        rr.g = 12.92 * rr.g;
    }
    else
    {
        rr.g = 1.055 * pow(rr.g, 1.0/2.4) - 0.055;
    }

    if(b < 0.0031308)
    {
        rr.b = 12.92 * rr.b;
    }
    else
    {
        rr.b = 1.055 * pow(rr.b, 1.0/2.4) - 0.055;
    }

    
    return rr;


}

var<private> seed : vec4<u32> = vec4<u32>(0);







