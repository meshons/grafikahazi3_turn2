//=============================================================================================
// Mintaprogram: Zold haromszog. Ervenyes 2018. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Stork Gabor
// Neptun : NO047V
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

// source: http://cg.iit.bme.hu/portal/sites/default/files/oktatott%20t%C3%A1rgyak/sz%C3%A1m%C3%ADt%C3%B3g%C3%A9pes%20grafika/inkrement%C3%A1lis%203d%20k%C3%A9pszint%C3%A9zis/3dendzsinke.cpp

struct Material {
    vec3 kd, ks;
    float shininess;
};

vec4 qmul(vec4 q1, vec4 q2) {
    vec3 d1(q1.x, q1.y, q1.z), d2(q2.x, q2.y, q2.z);
    vec3 helper = d2 * q1.w + d1 * q2.w + cross(d1, d2);
    return vec4(helper.x, helper.y, helper.z, q1.w * q2.w - dot(d1, d2));
}

struct Light {
    vec3 Le;
    vec4 wLightPos;
    vec4 animationCenter;

    void animate(float t) {
        t/=1000;
        vec3 position = vec3(wLightPos.x, wLightPos.y, wLightPos.z);
        vec3 origo = vec3(animationCenter.x, animationCenter.y, animationCenter.z);
        vec3 point = position - origo;
        vec4 q = vec4(
                sinf(t/4) * cosf(t) / 2,
                sinf(t/4) * sinf(t) / 2,
                sinf(t/4) * sqrtf(3 / 4),
                cos(t/4)
        );
        float qAbs = sqrtf(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
        vec4 qInv = vec4(
                -1 * q.x,
                -1 * q.y,
                -1 * q.z,
                q.w
                );
        qInv = qInv / (qAbs * qAbs);
        vec4 qr = qmul(qmul(q, vec4(point.x, point.y, point.z, 0)), qInv);
        qr.w = 1;
        wLightPos = qr;
    }
};

struct RenderState {
    mat4	           MVP, M, Minv, V, P;
    Material *         material;
    std::vector<Light> lights;
    vec3	           wEye;
    bool               animation;
};

class Shader : public GPUProgram {
public:
    virtual void Bind(RenderState state) = 0;

    void setUniformMaterial(const Material& material, const std::string& name) {
        setUniform(material.kd, name + ".kd");
        setUniform(material.ks, name + ".ks");
        setUniform(material.shininess, name + ".shininess");
    }

    void setUniformLight(const Light& light, const std::string& name) {
        setUniform(light.Le, name + ".Le");
        setUniform(light.wLightPos, name + ".wLightPos");
    }
};

class RouteShader : public Shader {
    const char * vertexSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 Le;
			vec4 wLightPos;
		};

		uniform mat4  MVP, M, Minv;
		uniform Light[2] lights;
		uniform vec3  wEye;
        uniform bool animation;

		layout(location = 0) in vec3  vtxPos;
		layout(location = 1) in vec3  vtxNorm;

		out vec3 wNormal;
		out vec3 wView;
		out vec3 wLight[2];

		void main() {
            vec3 vtxPos2 = vtxPos;
            if (!animation)
                vtxPos2.z = 0.02;

			gl_Position = vec4(vtxPos2, 1) * MVP;

			vec4 wPos = vec4(vtxPos2, 1) * M;
			for(int i = 0; i < 2; i++) {
				wLight[i] = lights[i].wLightPos.xyz * wPos.w - wPos.xyz * lights[i].wLightPos.w;
			}
		    wView  = wEye * wPos.w - wPos.xyz;
		    wNormal = (Minv * vec4(vtxNorm, 0)).xyz;
		}
	)";

    const char * fragmentSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 Le;
			vec4 wLightPos;
		};

		struct Material {
			vec3 kd, ks;
			float shininess;
		};

		uniform Material material;
		uniform Light[2] lights;

		in  vec3 wNormal;
		in  vec3 wView;
		in  vec3 wLight[2];

        out vec4 fragmentColor;

        vec3 Fresnel(vec3 F0, float cosTheta) {
            return F0 + (vec3(1, 1, 1) - F0) * pow(cosTheta, 5);
        }

		void main() {
			vec3 N = normalize(wNormal);
			vec3 V = normalize(wView);
			//if (dot(N, V) < 0) N = -N;
			vec3 kd = material.kd;

			vec3 radiance = vec3(0, 0, 0);
			for(int i = 0; i < 2; i++) {
				vec3 L = normalize(wLight[i]);
				vec3 H = normalize(L + V);
				float cost = max(dot(N,L), 0), cosd = max(dot(N,H), 0);
				radiance += kd * lights[i].Le + (material.ks * pow(cosd, material.shininess)) * lights[i].Le;
			}
			fragmentColor = vec4(radiance, 1);
		}
	)";
public:
    RouteShader() {
        create(
                vertexSource,
                fragmentSource,
                "fragmentColor"
                );
    }

    void Bind(RenderState state) {
        Use();
        setUniform(state.MVP, "MVP");
        setUniform(state.M, "M");
        setUniform(state.Minv, "Minv");
        setUniform(state.wEye, "wEye");
        setUniformMaterial(*state.material, "material");

        setUniform(state.animation, "animation");
        for (unsigned int i = 0; i < state.lights.size(); i++) {
            setUniformLight(state.lights[i], std::string("lights[") + std::to_string(i) + std::string("]"));
        }
    }
};

class TerrainShader : public Shader {
    const char * vertexSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 Le;
			vec4 wLightPos;
		};

		uniform mat4  MVP, M, Minv;
		uniform Light[2] lights;
		uniform vec3  wEye;
        uniform bool animation;

		layout(location = 0) in vec3  vtxPos;
		layout(location = 1) in vec3  vtxNorm;

		out vec3 wNormal;
		out vec3 wView;
		out vec3 wLight[2];
        out float height;

		void main() {
            height = (vtxPos.z + 2) / 4;
            vec3 vtxPos2 = vtxPos;
            if (!animation)
                vtxPos2.z = 0;

			gl_Position = vec4(vtxPos2, 1) * MVP;

			vec4 wPos = vec4(vtxPos2, 1) * M;
			for(int i = 0; i < 2; i++) {
				wLight[i] = lights[i].wLightPos.xyz * wPos.w - wPos.xyz * lights[i].wLightPos.w;
			}
		    wView  = wEye * wPos.w - wPos.xyz;
		    wNormal = (Minv * vec4(vtxNorm, 0)).xyz;
		}
	)";

    const char * fragmentSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 Le;
			vec4 wLightPos;
		};

		struct Material {
			vec3 kd, ks;
			float shininess;
		};

		uniform Material material;
		uniform Light[2] lights;

        const vec3 green = vec3(0.105, 0.76, 0.286);
        const vec3 brown = vec3(0.301, 0.1529, 0.1372);

		in  vec3 wNormal;
		in  vec3 wView;
		in  vec3 wLight[2];
        in  float height;

        out vec4 fragmentColor;

		void main() {
			vec3 N = normalize(wNormal);
			vec3 V = normalize(wView);

			vec3 kd = material.kd * (green * (1 - height) + brown * height);

			vec3 radiance = vec3(0, 0, 0);
			for(int i = 0; i < 2; i++) {
				vec3 L = normalize(wLight[i]);
				vec3 H = normalize(L + V);
				float cost = max(dot(N,L), 0), cosd = max(dot(N,H), 0);
				radiance += (kd * cost + material.ks * pow(cosd, material.shininess)) * lights[i].Le;
			}
			fragmentColor = vec4(radiance, 1);
		}
	)";
public:
    TerrainShader() {
        create(
                vertexSource,
                fragmentSource,
                "fragmentColor"
        );
    }

    void Bind(RenderState state) {
        Use();
        setUniform(state.MVP, "MVP");
        setUniform(state.M, "M");
        setUniform(state.Minv, "Minv");
        setUniform(state.wEye, "wEye");
        setUniform(state.animation, "animation");
        setUniformMaterial(*state.material, "material");

        for (unsigned int i = 0; i < state.lights.size(); i++) {
            setUniformLight(state.lights[i], std::string("lights[") + std::to_string(i) + std::string("]"));
        }
    }
};

struct VertexData {
    vec3 position, normal;
};

class Geometry {
protected:
    unsigned int vao, vbo;
public:
    Geometry() {
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);
        glGenBuffers(1, &vbo);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
    }
    virtual void draw() = 0;
    ~Geometry() {
        glDeleteBuffers(1, &vbo);
        glDeleteVertexArrays(1, &vao);
    }
};

const int tessellationLevel = 50;

class ParamSurface : public Geometry {
protected:
    unsigned int nVtxPerStrip, nStrips;
public:
    ParamSurface() { nVtxPerStrip = nStrips = 0; }

    virtual VertexData genVertexData(float u, float v) = 0;

    void create(int N = tessellationLevel, int M = tessellationLevel) {
        nVtxPerStrip = (M + 1) * 2;
        nStrips = N;
        std::vector<VertexData> vtxData;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j <= M; j++) {
                vtxData.push_back(genVertexData((float) j / M, (float) i / N));
                vtxData.push_back(genVertexData((float) j / M, (float) (i + 1) / N));
            }
        }
        glBufferData(GL_ARRAY_BUFFER, nVtxPerStrip * nStrips * sizeof(VertexData), &vtxData[0], GL_STATIC_DRAW);

        glEnableVertexAttribArray(0);
        glEnableVertexAttribArray(1);

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, position));
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, normal));
    }

    void draw() {
        glBindVertexArray(vao);
        for (unsigned int i = 0; i < nStrips; i++)
            glDrawArrays(GL_TRIANGLE_STRIP, i *  nVtxPerStrip, nVtxPerStrip);
    }
};

class Terrain : public ParamSurface {
public:
    Terrain() { create(); }

    VertexData genVertexData(float u, float v) {
        VertexData vd;
        u *= 10;
        v *= 10;
        vd.position = {u, v, h(u, v)};
        vd.normal = normal(u, v);
        return vd;
    }

    float h(float u, float v) {
        float h = cosf(-1 * u + 0.3f * v + 7.0f / 5.0f)
                + cosf(0.3f * u + v - 13.0f / 5.0f);
        return h;
    }

    vec3 normal(float u, float v) {
        vec3 normal;
        normal.x = (-1.0f * sinf(-1 * u - 0.3f * v + 7.0f / 5.0f)
                + 0.3f * sinf(0.3f * u + v - 13.0f / 5.0f));
        normal.y = (0.3f * sinf(-1 * u + 0.3f * v + 7.0f / 5.0f)
                + sinf(0.3f * u + v - 13.0f / 5.0f));
        normal.z = 1;
        return normalize(normal);
    }
};

class Route : public ParamSurface {
    std::vector<vec2> cps;

    Terrain * terrain;

    vec2 v(const vec2 & p1, float t1, const vec2 & p2, float t2, const vec2 & p3, float t3) const {
        return ((p3 - p2) * (1 / (t3 - t2)) + ((p2 - p1) * (1 / (t2 - t1)))) * 0.5f;
    }

    int posCalculator(int pos) const {
        while (pos < 0)
            pos += cps.size() - 1;
        while (pos >= cps.size())
            pos -= cps.size() - 1;
        return pos;
    }

    vec2 rHelper(
            const vec2 & p1, float t1, const vec2 & p2, float t2, const vec2 & p3, float t3, const vec2 & p4, float t4,
            float t
    ) const {
        vec2 v2 = v(p1, t1, p2, t2, p3, t3);
        vec2 v3 = v(p2, t2, p3, t3, p4, t4);

        vec2 a0 = p2;
        vec2 a1 = v2;
        vec2 a2 = (((p3 - p2) * 3.0f) * (1 / ((t3 - t2) * ((t3 - t2))))) - ((v3 + v2 * 2.0f) * (1 / (t3 - t2)));
        vec2 a3 = (((p2 - p3) * 2.0f) * (1 / ((t3 - t2) * (t3 - t2) * (t3 - t2)))) +
                  ((v3 + v2) * (1 / ((t3 - t2) * (t3 - t2))));

        float tt0 = t - t2;
        float tt1 = tt0 * tt0;
        float tt2 = tt1 * tt0;

        return {a3 * tt2 + a2 * tt1 + a1 * tt0 + a0};
    }

    vec2 rDerivativeHelper(
            const vec2 & p1, float t1, const vec2 & p2, float t2, const vec2 & p3, float t3, const vec2 & p4, float t4,
            float t
    ) const {
        vec2 v2 = v(p1, t1, p2, t2, p3, t3);
        vec2 v3 = v(p2, t2, p3, t3, p4, t4);

        vec2 a1 = v2;
        vec2 a2 = (((p3 - p2) * 3.0f) * (1 / ((t3 - t2) * ((t3 - t2))))) - ((v3 + v2 * 2.0f) * (1 / (t3 - t2)));
        vec2 a3 = (((p2 - p3) * 2.0f) * (1 / ((t3 - t2) * (t3 - t2) * (t3 - t2)))) +
                  ((v3 + v2) * (1 / ((t3 - t2) * (t3 - t2))));

        float tt0 = t - t2;
        float tt1 = tt0 * tt0;

        return {3 * a3 * tt1 + 2 * a2 * tt0 + a1};
    }

    vec2 r(float t) {
        if (cps.size() < 2)
            return {0, 0};
        cps.push_back(cps[0]);
        int pos = t * (float)(cps.size() - 1);
        vec2 returnValue = rHelper(
                cps[posCalculator(pos - 1)],
                ((float)pos - 1) / (float)(cps.size() - 1),
                cps[posCalculator(pos)],
                ((float)pos) / (float)(cps.size() - 1),
                cps[posCalculator(pos + 1)],
                ((float)pos + 1) / (float)(cps.size() - 1),
                cps[posCalculator(pos + 2)],
                ((float)pos + 2) / (float)(cps.size() - 1),
                t
                );
        cps.pop_back();
        return returnValue;
    }

    vec2 rDerivative(float t) {
        if (cps.size() < 2)
            return {0, 0};
        cps.push_back(cps[0]);
        int pos = t * (float)(cps.size() - 1);
        vec2 returnValue = rDerivativeHelper(
                cps[posCalculator(pos - 1)],
                ((float)pos - 1) / (float)(cps.size() - 1),
                cps[posCalculator(pos)],
                ((float)pos) / (float)(cps.size() - 1),
                cps[posCalculator(pos + 1)],
                ((float)pos + 1) / (float)(cps.size() - 1),
                cps[posCalculator(pos + 2)],
                ((float)pos + 2) / (float)(cps.size() - 1),
                t
        );
        cps.pop_back();
        return normalize(returnValue);
    }

    const float h0 = 0.01f;
    const float r0 = 0.01f;

public:
    float lengthOfRoute = 0;

    vec3 s(float t) {
        vec3 position;
        vec2 position2d = r(t);
        position.x = position2d.x;
        position.y = position2d.y;
        position.z = terrain->h(position.x + 5, position.y + 5) + h0;
        return position;;
    }

    vec3 tangent(float t) {
        vec2 positionDerivative = rDerivative(t);
        vec3 tangentVector;
        vec2 position = r(t);
        tangentVector.x = positionDerivative.x;
        tangentVector.y = positionDerivative.y;
        vec3 terrainNormal = terrain->normal(position.x + 5, position.y + 5);
        tangentVector.z = -1 * terrainNormal.x * positionDerivative.x + -1 * terrainNormal.y * positionDerivative.y;
        return normalize(tangentVector);
    }

    vec3 normal(float t) {
        vec2 position = r(t);
        return normalize(terrain->normal(position.x + 5, position.y + 5));
    }

    vec3 biNormal(float t) {
        return normalize(cross(normal(t), tangent(t)));
    }

    float X(float v) {
        return sinf(v * 2 * (float)M_PI) * r0;
    }

    float Y(float v) {
        return cosf(v * 2 * (float)M_PI) * r0;
    }

public:
    vec3 p(float u, float v) {
        vec3 position;
        position = biNormal(u) * X(v) + normal(u) * Y(v);
        return position;
    }

    Route(Terrain * _terrain) :
        terrain{_terrain} { create(10, 250); }

    void addControlPoint(vec2 cp) {
        cps.push_back(cp);
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);

        int N = 10, M = 250;
        nVtxPerStrip = (M + 1) * 2;
        nStrips = N;
        std::vector<VertexData> vtxData;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j <= M; j++) {
                vtxData.push_back(genVertexData((float) j / M, (float) i / N));
                vtxData.push_back(genVertexData((float) j / M, (float) (i + 1) / N));
            }
        }
        lengthOfRoute = 0;
        for (int i = 0; i < M; i++)
            lengthOfRoute += length(s((float) i / M) - s((float) (i + 1) / M));
        glBufferData(GL_ARRAY_BUFFER, nVtxPerStrip * nStrips * sizeof(VertexData), &vtxData[0], GL_STATIC_DRAW);
    }

    VertexData genVertexData(float u, float v) {
        VertexData vd;
        vd.position = s(u) + p(u, v);
        vd.normal = normalize(
                    vd.position - s(u)
                );
        return vd;
    }

    void draw() {
        glBindVertexArray(vao);

        for (unsigned int i = 0; i < nStrips; i++)
            glDrawArrays(GL_TRIANGLE_STRIP, i *  nVtxPerStrip, nVtxPerStrip);
    }
};

struct Object {
    Shader *   shader;
    Material * material;
    Geometry * geometry;
    vec3 translation;
public:
    Object(Shader * _shader, Material * _material, Geometry * _geometry) :
    translation(vec3(0, 0, 0)) {
        shader = _shader;
        material = _material;
        geometry = _geometry;
    }
    virtual void setModelingTransform(mat4& M, mat4& Minv) {
        M = TranslateMatrix(translation);
        Minv = TranslateMatrix(-translation);
    }

    void draw(RenderState state) {
        mat4 M, Minv;
        setModelingTransform(M, Minv);
        state.M = M;
        state.Minv = Minv;
        state.MVP = state.M * state.V * state.P;
        state.material = material;
        shader->Bind(state);
        geometry->draw();
    }

    virtual void animate(float tstart, float tend) {}
};

vec3 vec3mat4(const vec3 & v, const mat4 & m) {
    vec4 v4 = vec4(v.x, v.y, v.z, 1);
    v4 = v4 * m;
    return vec3(v4.x, v4.y, v4.z);
}

struct Camera
{
    vec3 wEye, wLookat, wVup;
    float fov, asp, fp, bp;
    Route * route;
    Object * routeObject;
public:
    Camera(Route * _route, Object * _routeObject) : route{_route}, routeObject{_routeObject} {
        asp = (float) windowWidth / windowHeight;
        fov = 75.0f * (float) M_PI / 180.0f;
        fp = 0.0001;
        bp = 100;
    }

    mat4 V() {
        vec3 w = normalize(wEye - wLookat);
        vec3 u = normalize(cross(wVup, w));
        vec3 v = cross(w, u);
        return TranslateMatrix(wEye * (-1)) * mat4(
                u.x, v.x, w.x, 0,
                u.y, v.y, w.y, 0,
                u.z, v.z, w.z, 0,
                0, 0, 0, 1
        );
    }

    mat4 P() {
        return mat4(
                1 / (tan(fov / 2) * asp), 0, 0, 0,
                0, 1 / tan(fov / 2), 0, 0,
                0, 0, -(fp + bp) / (bp - fp), -1,
                0, 0,-2 * fp * bp / (bp - fp), 0
        );
    }

    void animate(float t) {
        float totalDistance = route->lengthOfRoute * 10;
        float totalTime = totalDistance / 20;
        const float l = 0.02;
        t = fmod(t, totalTime) / totalTime;
        mat4 M, MInv;
        routeObject->setModelingTransform(M, MInv);
        wEye = vec3mat4(route->s(t) + route->normal(t) * l, M);
        wLookat = wEye + route->tangent(t);
        wVup = route->normal(t);
    }
};

class Scene {
    std::vector<Object *> objects;
    Camera * camera;
    bool animation = false;
    std::vector<Light> lights;
    std::vector<Shader *> shaders;
    std::vector<Material *> materials;
    Terrain * terrain;
    Route * route;

    bool built = false;
public:
    void build() {
        shaders.push_back(new TerrainShader());
        shaders.push_back(new RouteShader());

        materials.push_back(new Material);
        materials[0]->kd = vec3(1, 1, 1);
        materials[0]->ks = vec3(0.1, 0.1, 0.1);
        materials[0]->shininess = 10;

        materials.push_back(new Material);
        materials[1]->kd = vec3(0.14f, 0.16f, 0.13f);
        materials[1]->ks = vec3(4.1f, 2.3f, 3.1f);
        materials[1]->shininess = 100;

        terrain = new Terrain();
        route = new Route(terrain);

        Object * terrainObject = new Object(shaders[0], materials[0], terrain);
        terrainObject->translation = vec3(-5, -5, 0);
        objects.push_back(terrainObject);

        Object * routeObject = new Object(shaders[1], materials[1], route);
        routeObject->translation = vec3( 0, 0, 0);
        objects.push_back(routeObject);

        camera = new Camera(route, routeObject);

        camera->wEye = vec3(0, 0, 6);
        camera->wLookat = vec3(0, 0, 0);
        camera->wVup = vec3(0, 1, 0);

        lights.resize(2);
        lights[0].wLightPos = vec4(0, 0,  3, 0);
        lights[0].Le = vec3(0.7, 0.7, 0.7);

        lights[1].wLightPos = vec4(0, 0, -5, 0);
        lights[1].Le = vec3(0.7, 0.7, 0.7);

        lights[0].animationCenter = lights[1].wLightPos;
        lights[1].animationCenter = lights[0].wLightPos;

        built = true;
    }

    ~Scene() {
        if (built) {
            delete camera;
            for (int i = 0; i < objects.size(); ++i)
                delete objects[i];
            for (int i = 0; i < shaders.size(); ++i)
                delete shaders[i];
            delete route;
            delete terrain;
            for (int i = 0; i < materials.size(); ++i)
                delete materials[i];
        }
    }

    void render() {
        RenderState state;
        state.wEye = camera->wEye;
        state.V = camera->V();
        state.P = camera->P();
        state.lights = lights;
        state.animation = animation;
        for (Object * obj : objects)
            obj->draw(state);
    }

    void startAnimation() {
        animation = true;
    }

    void animate(float tstart, float tend) {
        if (animation) camera->animate(tend);
        for (unsigned int i = 0; i < lights.size(); i++) {
            lights[i].animate(tend);
        }
    }

    void addRoutePoint(vec2 && pt) {
        if (!animation)
            route->addControlPoint(pt);
    }
};

Scene scene;

void onInitialization() {
    glViewport(0, 0, windowWidth, windowHeight);
    glEnable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    scene.build();
}

void onDisplay() {
    glClearColor(	0, 0, 0, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    scene.render();
    glutSwapBuffers();
}

void onKeyboard(unsigned char key, int pX, int pY) {
    scene.startAnimation();
}

void onKeyboardUp(unsigned char key, int pX, int pY) {
    scene.startAnimation();
}

void onMouseMotion(
        int pX, int pY
) {
}

void onMouse(
        int button, int state, int pX, int pY
) {
    float cX = 2.0f * pX / windowWidth - 1;
    float cY = 1.0f - 2.0f * pY / windowHeight;

    switch (button) {
        case GLUT_LEFT_BUTTON:
            if (state == GLUT_DOWN)
                scene.addRoutePoint({cX * 5, cY * 5});
            break;
    }
}

void onIdle() {
    static float tend = 0;
    const float dt = 0.1f;
    float tstart = tend;
    tend = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;

    for (float t = tstart; t < tend; t += dt) {
        float Dt = fmin(dt, tend - t);
        scene.animate(t, t + Dt);
    }
    glutPostRedisplay();
}
