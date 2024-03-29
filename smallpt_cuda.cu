#include <cmath>    // smallpt, a Path Tracer by Kevin Beason, 2008
#include <cstdlib>  // Make : g++ -O3 -fopenmp smallpt.cpp -o smallpt
#include <cstdio>   //        Remove "-fopenmp" for g++ version < 4.2
#include <ctime>
#include <algorithm>
#include <numbers>

#define USE_CUDA
// #define USE_HIP

#if defined(USE_CUDA)
#include "cuda.h"
// #include "cuda_runtime.h"
#include "curand.h"
#include "curand_kernel.h"
#include "device_launch_parameters.h"
#elif defined(USE_HIP)
// TODO
#endif

#if defined(USE_CUDA) || defined(USE_HIP)
    #define GPU_RENDER

    #define HOST __host__
    #define ENTRY __global__
    #define KERNEL __host__ __device__
    #define GPU __device__
    #define CONST __constant__

    #if defined(__CUDA_ARCH__) || defined(__HIP__)
        #define GPU_CODE
    #endif
#else
    #define CPU_RENDER

    #define HOST
    #define ENTRY
    #define KERNEL
    #define GPU
    #define CONST
#endif

constexpr double Pi = 3.141592653589793;

// struct RandomLCG
// {
//     unsigned mSeed;

//     KERNEL RandomLCG(unsigned seed = 0) : mSeed(seed) {}
//     // KERNEL double operator()() { mSeed = 214013 * mSeed + 2531011; return mSeed * (1.0 / 4294967296); }
//     KERNEL double operator()() {}
// };
GPU __forceinline__
double random_double(curandStateXORWOW_t* rand_state){
    return curand_uniform_double(rand_state);
}

struct Vector3
{                   // Usage: time ./smallpt 5000 && xv image.ppm
    double x, y, z; // position, also color (r,g,b)

    KERNEL constexpr Vector3(double x_ = 0, double y_ = 0, double z_ = 0):
        x{ x_ }, y{ y_ }, z{ z_ }
    {
    }

    KERNEL Vector3 operator-() const { return Vector3(-x, -y, -z); }

    KERNEL Vector3 operator+(const Vector3& b) const { return Vector3(x + b.x, y + b.y, z + b.z); }
    KERNEL Vector3 operator-(const Vector3& b) const { return Vector3(x - b.x, y - b.y, z - b.z); }
    KERNEL Vector3 operator*(double b) const { return Vector3(x * b, y * b, z * b); }
    KERNEL Vector3 operator/(double b) const { return Vector3(x / b, y / b, z / b); }

    // only for Color
    KERNEL Vector3 operator*(const Vector3& b) const { return Vector3(x * b.x, y * b.y, z * b.z); }

    KERNEL Vector3& Normalize() { return *this = *this * (1 / sqrt(x * x + y * y + z * z)); }

    KERNEL double Dot(const Vector3& b) const { return x * b.x + y * b.y + z * b.z; }
    KERNEL Vector3 Cross(Vector3& b) { return Vector3(y * b.z - z * b.y, z * b.x - x * b.z, x * b.y - y * b.x); }

    KERNEL friend Vector3 operator*(double b, Vector3 v) { return v * b; }
    KERNEL friend double Dot(const Vector3& a, const Vector3& b) { return a.Dot(b); }
};

using Float3 = Vector3;
using Point3 = Vector3;
using Normal3 = Vector3;
using UnitVector3 = Vector3;
using Color = Vector3;

struct Ray
{
    Point3 origin;
    UnitVector3 direction;

    KERNEL Ray(Point3 origin_, UnitVector3 direction_): origin(origin_), direction(direction_) {}
};

enum class MaterialType
{
    Diffuse,
    Specular,
    Refract
}; // material types, used in radiance()

struct Sphere
{
    double radius;
    Point3 center;

    Color emission; // for area light
    Color color; // surface albedo, per-component is surface reflectance
    MaterialType materialType;

    KERNEL constexpr Sphere(double radius_, Vector3 center_, Color emission_, Color color_, MaterialType materialType):
        radius(radius_), center(center_), emission(emission_), color(color_), materialType(materialType) {}

    KERNEL double Intersect(const Ray& ray) const
    {
        // returns distance, 0 if nohit

        Vector3 oc = center - ray.origin;
        double neg_b = oc.Dot(ray.direction);
        double det = neg_b * neg_b - oc.Dot(oc) + radius * radius;

        if (det < 0)
            return 0;
        else
            det = sqrt(det);

        double epsilon = 1e-4;
        if (neg_b - det > epsilon)
        {
            return neg_b - det;
        }
        else if (neg_b + det > epsilon)
        {
            return neg_b + det;
        }

        return 0;
    }
};

CONST constexpr Sphere Scene[] =
{
    //Scene: radius, center, emission, albedo, material
    Sphere(1e5, Vector3(1e5 - 50, 0, 0),   Color(), Color(.6, .05, .05), MaterialType::Diffuse), //Left
    Sphere(1e5, Vector3(-1e5 + 50, 0, 0), Color(), Color(.15, .45, .1), MaterialType::Diffuse), //Right
    Sphere(1e5, Vector3(0, 0, 1e5 - 81.6),         Color(), Color(.75, .75, .75), MaterialType::Diffuse), //Back
    Sphere(1e5, Vector3(0, 0, -1e5 + 88.4),  Color(), Color(.75, .75, .75),              MaterialType::Diffuse), //Front
    Sphere(1e5, Vector3(0, 1e5 - 50, 0),         Color(), Color(.75, .75, .75), MaterialType::Diffuse), //Bottom
    Sphere(1e5, Vector3(0, -1e5 +49, 0), Color(), Color(.75, .75, .75), MaterialType::Diffuse), //Top

    Sphere(20, Vector3(-20, -30, -40),          Color(), Color(1, 1, 1), MaterialType::Specular), //Mirror
    Sphere(20, Vector3(20, -30, 0),          Color(), Color(1, 1, 1), MaterialType::Refract),  //Glass
    Sphere(600,  Vector3(0, 649 - .4, -20), Color(20, 20, 20), Color(),     MaterialType::Diffuse)   //Light
};

KERNEL inline double Lerp(double a, double b, double t) { return a + t * (b - a); }
KERNEL inline double Clamp(double x) { return x < 0 ? 0 : x > 1 ? 1 : x; }
inline int GammaEncoding(double x) { return int(pow(Clamp(x), 1 / 2.2) * 255 + .5); }

KERNEL inline bool Intersect(const Ray& ray, double& minDistance, int& id)
{
    double infinity = 1e20;
    minDistance = infinity;

    int sphereNum = sizeof(Scene) / sizeof(Sphere);
    double distance{};

    for (int i = sphereNum; i--;)
    {
        if ((distance = Scene[i].Intersect(ray)) && distance < minDistance)
        {
            minDistance = distance;
            id = i;
        }
    }

    return minDistance < infinity;
}

GPU Color Radiance(const Ray& ray, int depth, curandStateXORWOW_t& rng)
{
    double distance; // distance to intersection
    int id = 0; // id of intersected object

    if (!Intersect(ray, distance, id))
        return Color(); // if miss, return black

    const Sphere& obj = Scene[id]; // the hit object

    if (depth > 5)
        return obj.emission; // if path is too long, only return sphere's emission

    // intersection property
    Vector3 position = ray.origin + ray.direction * distance;
    Normal3 normal = (position - obj.center).Normalize();
    Normal3 shading_normal = normal.Dot(ray.direction) < 0 ? normal : normal * -1;

    Color f = obj.color; // this is surface albedo $\rho$
    double max_component = (f.x > f.y && f.x > f.z) ? f.x : (f.y > f.z ? f.y : f.z); // max refl

    //russian roulette
    if (++depth > 3)
    {
        if (random_double(&rng) < max_component)
            f = f * (1 / max_component);
        else
            return obj.emission;
    }

    if (obj.materialType == MaterialType::Diffuse) // Ideal Diffuse reflection
    {
        double random1 = 2 * Pi * random_double(&rng);
        double random2 = random_double(&rng);
        double random2Sqrt = sqrt(random2);

        // shading coordinate on intersection
        Vector3 w = shading_normal;
        Vector3 u = ((fabs(w.x) > .1 ? Vector3(0, 1, 0) : Vector3(1, 0, 0)).Cross(w)).Normalize();
        Vector3 v = w.Cross(u);

        // Cosine importance sampling of the hemisphere for diffuse reflection
        Vector3 direction = (u * cos(random1) * random2Sqrt + v * sin(random1) * random2Sqrt + w * sqrt(1 - random2)).Normalize();

        f = f / Pi; // for lambert brdf, f = R / Pi;
        double abs_cos_theta = std::abs(shading_normal.Dot(direction));
        double pdf = abs_cos_theta / Pi; // cosine-weighted sampling
        return obj.emission + (f * Radiance(Ray(position, direction), depth, rng) * abs_cos_theta) / pdf;
    }
    else if (obj.materialType == MaterialType::Specular) // Ideal Specular reflection
    {
        Vector3 direction = ray.direction - normal * 2 * normal.Dot(ray.direction);
        return obj.emission + f * Radiance(Ray(position, direction), depth, rng);
    }
    else // Ideal Dielectric Refraction
    {
        bool into = normal.Dot(shading_normal) > 0; // Ray from outside going in?

        // IOR(index of refractive)
        double etaI = 1; // vacuum
        double etaT = 1.5; // glass
        double eta = into ? etaI / etaT : etaT / etaI;


        // compute reflect direction by refection law
        Ray reflectRay(position, ray.direction - normal * 2 * normal.Dot(ray.direction));

        // compute refract direction by Snell's law
        // https://www.pbr-book.org/3ed-2018/Reflection_Models/Specular_Reflection_and_Transmission#SpecularTransmission see `Refract()`
        double cosThetaI = ray.direction.Dot(shading_normal);
        double cosThetaT2 = cosThetaT2 = 1 - eta * eta * (1 - cosThetaI * cosThetaI);
        if (cosThetaT2 < 0) // Total internal reflection
            return obj.emission + f * Radiance(reflectRay, depth, rng);
        double cosThetaT = sqrt(cosThetaT2);
        Vector3 refractDirection = (ray.direction * eta - normal * ((into ? 1 : -1) * (cosThetaI * eta + cosThetaT))).Normalize();


        // compute the fraction of incoming light that is reflected or transmitted
        // by Schlick Approximation of Fresnel Dielectric 1994 https://en.wikipedia.org/wiki/Schlick%27s_approximation
        double a = etaT - etaI;
        double b = etaT + etaI;
        double R0 = a * a / (b * b);
        double c = 1 - (into ? -cosThetaI : refractDirection.Dot(normal));

        double Re = R0 + (1 - R0) * c * c * c * c * c;
        double Tr = 1 - Re;


        // probability of reflected or transmitted
        double P = .25 + .5 * Re;
        double RP = Re / P;
        double TP = Tr / (1 - P);

        Color Li;
        if (depth > 2)
        {
            // Russian roulette
            if (random_double(&rng) < P)
            {
                Li = Radiance(reflectRay, depth, rng) * RP;
            }
            else
            {
                Li = Radiance(Ray(position, refractDirection), depth, rng) * TP;
            }
        }
        else
        {
            Li = Radiance(reflectRay, depth, rng) * Re + Radiance(Ray(position, refractDirection), depth, rng) * Tr;
        }

        return obj.emission + f * Li;
    }
}



#if defined(GPU_RENDER)

#if defined(USE_CUDA)

void CheckCUDA(cudaError_t result, const char* const file, int const line, const char* const func)
{
    if (result != cudaSuccess)
    {
        cudaError_t error = cudaGetLastError();
        std::fprintf(stderr, "CUDA error: %s, at %s:%d `%s`", cudaGetErrorString(error), file, line, func);

        cudaDeviceReset();
        exit(99);
    }
}

#define CHECK_CUDA(val) CheckCUDA((val), __FILE__, __LINE__,  #val)

ENTRY void Kernel(Color* film, int width, int height, int samplesPerPixel)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if ((x >= width) || (y >= height))
        return;

    double offset = x + y * blockDim.x * gridDim.x;
    curandStateXORWOW_t rng;
    curand_init(offset, 0, 0, &rng);

    // TODO
    Ray camera(Vector3(0, 0, 220), Vector3(0, 0, -1)); // camera posotion, direction
    Vector3 cx = Vector3(width * .5 / height); // left
    Vector3 cy = (cx.Cross(camera.direction)).Normalize() * .5; // up

    Color color;
    for (int s = 0; s < samplesPerPixel; ++s)
    {
        // RandomLCG rng(y * width + x * samplesPerPixel + s);
        Vector3 direction = cx * ((x + s / (float) samplesPerPixel) / width - .5) +
                            cy * ((y + s / (float) samplesPerPixel) / height - .5) + camera.direction;

        color = color + Radiance(Ray(camera.origin + direction * 140, direction.Normalize()), 0, rng);
    }
    color = color * (1. / samplesPerPixel);

    int index = (height - y - 1) * width + x;
    film[index] = Color(Clamp(color.x), Clamp(color.y), Clamp(color.z));
}

class Device
{
public:
    Color* Render(int width, int height, int samplesPerPixel)
    {
        // TODO
        size_t stackSize{};
        CHECK_CUDA(cudaDeviceGetLimit(&stackSize, cudaLimitStackSize));
        printf("stack size: %d\n", stackSize);
        CHECK_CUDA(cudaDeviceSetLimit(cudaLimitStackSize, stackSize * 3));

        Color* film;
        size_t film_size = width * height * 3 * sizeof(double);
        CHECK_CUDA(cudaMallocManaged((void**)&film, film_size));

        int tx = 8;
        int ty = 8;
        dim3 blocks(width / tx + 1, height / ty + 1);
        dim3 threads(tx, ty);
        Kernel<<<blocks, threads>>>(film, width, height, samplesPerPixel);

        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        return film;
    }

    ~Device()
    {
        cudaFree(film);
    }

private:
    Color* film;
};

#elif defined(USE_HIP) 
// TODO
#else
#error
#endif

#else // CPU_RENDER

class Device
{
public:
    Color* Render(int width, int height, int samplesPerPixel)
    {
        film = new Vector3[width * height];

        // right hand
        Ray camera(Vector3(50, 52, 295.6), Vector3(0, -0.042612, -1).Normalize()); // camera posotion, direction
        Vector3 cx = Vector3(width * .5135 / height); // left
        Vector3 cy = (cx.Cross(camera.direction)).Normalize() * .5135; // up

        #pragma omp parallel for schedule(dynamic, 1) // OpenMP
        for (int y = 0; y < height; y++) // Loop over image rows
        {
            fprintf(stderr, "\rRendering (%d spp) %5.2f%%", samplesPerPixel, 100. * y / (height - 1));

            for (unsigned short x = 0; x < width; x++) // Loop cols
            {
                Color Li{};
                for (int s = 0; s < samplesPerPixel; s++)
                {
                    RandomLCG rng(y * width + x * samplesPerPixel + s);
                    Vector3 direction =
                        cx * ((rng() + x) / width - .5) +
                        cy * ((rng() + y) / height - .5) + camera.direction;

                    Li = Li + Radiance(Ray(camera.origin + direction * 140, direction.Normalize()), 0, rng) * (1. / samplesPerPixel);
                } // Camera rays are pushed ^^^^^ forward to start in interior

                int i = (height - y - 1) * width + x;
                film[i] = film[i] + Vector3(Clamp(Li.x), Clamp(Li.y), Clamp(Li.z));
            }
        }

        return film;
    }

    ~Device()
    {
        if (film)
            delete[] film;
    }

private:
    Color* film;
};

#endif



int main(int argc, char* argv[])
{
    int width = 1024, height = 1024;
    int samplesPerPixel = argc == 2 ? atoi(argv[1]) / 4 : 100;
    
    Device device;
    clock_t start = clock();
    Color* film = device.Render(width, height, samplesPerPixel);
    printf("\n%f sec\n", (float)(clock() - start) / CLOCKS_PER_SEC);

    FILE* image;
    errno_t err = fopen_s(&image, "image_kernel.ppm", "w"); // Write image to PPM file.
    fprintf(image, "P3\n%d %d\n%d\n", width, height, 255);
    for (int i = 0; i < width * height; i++)
        fprintf(image, "%d %d %d\n", GammaEncoding(film[i].x), GammaEncoding(film[i].y), GammaEncoding(film[i].z));

    return 0;
}