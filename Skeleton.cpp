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


GPUProgram gpuProgram;
struct Material {
    vec3 kd, ks, ka;
    float shininess;
};

struct Light {
    vec3 La, Le;
    vec4 wLightPos;

    void animate(float t) {	}
};

struct RenderState {
    mat4	           MVP, M, Minv, V, P;
    Material *         material;
    std::vector<Light> lights;
    vec3	           wEye;
};

class Shader : public GPUProgram {
public:
    virtual void Bind(RenderState state) = 0;

    void setUniformMaterial(const Material& material, const std::string& name) {
        setUniform(material.kd, name + ".kd");
        setUniform(material.ks, name + ".ks");
        setUniform(material.ka, name + ".ka");
        setUniform(material.shininess, name + ".shininess");
    }

    void setUniformLight(const Light& light, const std::string& name) {
        setUniform(light.La, name + ".La");
        setUniform(light.Le, name + ".Le");
        setUniform(light.wLightPos, name + ".wLightPos");
    }
};

//---------------------------
class RouteShader : public Shader {
    //---------------------------
    const char * vertexSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};

		uniform mat4  MVP, M, Minv;
		uniform Light[2] lights;
		uniform vec3  wEye;

		layout(location = 0) in vec3  vtxPos;
		layout(location = 1) in vec3  vtxNorm;

		out vec3 wNormal;
		out vec3 wView;
		out vec3 wLight[8];

		void main() {
			gl_Position = vec4(vtxPos, 1) * MVP; // to NDC
			// vectors for radiance computation
			vec4 wPos = vec4(vtxPos, 1) * M;
			for(int i = 0; i < 2; i++) {
				wLight[i] = lights[i].wLightPos.xyz * wPos.w - wPos.xyz * lights[i].wLightPos.w;
			}
		    wView  = wEye * wPos.w - wPos.xyz;
		    wNormal = (Minv * vec4(vtxNorm, 0)).xyz;
		}
	)";

    // fragment shader in GLSL
    const char * fragmentSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};

		struct Material {
			vec3 kd, ks, ka;
			float shininess;
		};

		uniform Material material;
		uniform Light[8] lights;    // light sources
		uniform int   nLights;

		in  vec3 wNormal;       // interpolated world sp normal
		in  vec3 wView;         // interpolated world sp view
		in  vec3 wLight[2];     // interpolated world sp illum dir

        out vec4 fragmentColor; // output goes to frame buffer

		void main() {
			vec3 N = normalize(wNormal);
			vec3 V = normalize(wView);
			if (dot(N, V) < 0) N = -N;	// prepare for one-sided surfaces like Mobius or Klein
			vec3 texColor = texture(diffuseTexture, texcoord).rgb;
			vec3 ka = material.ka * texColor;
			vec3 kd = material.kd * texColor;

			vec3 radiance = vec3(0, 0, 0);
			for(int i = 0; i < nLights; i++) {
				vec3 L = normalize(wLight[i]);
				vec3 H = normalize(L + V);
				float cost = max(dot(N,L), 0), cosd = max(dot(N,H), 0);
				// kd and ka are modulated by the texture
				radiance += ka * lights[i].La +
                           (kd * texColor * cost + material.ks * pow(cosd, material.shininess)) * lights[i].Le;
			}
			fragmentColor = vec4(radiance, 1);
		}
	)";
public:
    RouteShader() { create(vertexSource, fragmentSource, "fragmentColor"); }

    void Bind(RenderState state) {
        Use(); 		// make this program run
        setUniform(state.MVP, "MVP");
        setUniform(state.M, "M");
        setUniform(state.Minv, "Minv");
        setUniform(state.wEye, "wEye");
        setUniformMaterial(*state.material, "material");

        setUniform((int)state.lights.size(), "nLights");
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
			vec3 La, Le;
			vec4 wLightPos;
		};

		uniform mat4  MVP, M, Minv; // MVP, Model, Model-inverse
		uniform Light[2] lights;    // light sources
		uniform vec3  wEye;         // pos of eye

		layout(location = 0) in vec3  vtxPos;            // pos in modeling space
		layout(location = 1) in vec3  vtxNorm;      	 // normal in modeling space

		out vec3 wNormal;		    // normal in world space
		out vec3 wView;             // view in world space
		out vec3 wLight[8];		    // light dir in world space

		void main() {
			gl_Position = vec4(vtxPos, 1) * MVP; // to NDC
			// vectors for radiance computation
			vec4 wPos = vec4(vtxPos, 1) * M;
			for(int i = 0; i < 2; i++) {
				wLight[i] = lights[i].wLightPos.xyz * wPos.w - wPos.xyz * lights[i].wLightPos.w;
			}
		    wView  = wEye * wPos.w - wPos.xyz;
		    wNormal = (Minv * vec4(vtxNorm, 0)).xyz;
		}
	)";

    // fragment shader in GLSL
    const char * fragmentSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};

		struct Material {
			vec3 kd, ks, ka;
			float shininess;
		};

		uniform Material material;
		uniform Light[8] lights;    // light sources
		uniform int   nLights;

		in  vec3 wNormal;       // interpolated world sp normal
		in  vec3 wView;         // interpolated world sp view
		in  vec3 wLight[8];     // interpolated world sp illum dir

        out vec4 fragmentColor; // output goes to frame buffer

		void main() {
			vec3 N = normalize(wNormal);
			vec3 V = normalize(wView);
			if (dot(N, V) < 0) N = -N;	// prepare for one-sided surfaces like Mobius or Klein
			vec3 texColor = texture(diffuseTexture, texcoord).rgb;
			vec3 ka = material.ka * texColor;
			vec3 kd = material.kd * texColor;

			vec3 radiance = vec3(0, 0, 0);
			for(int i = 0; i < nLights; i++) {
				vec3 L = normalize(wLight[i]);
				vec3 H = normalize(L + V);
				float cost = max(dot(N,L), 0), cosd = max(dot(N,H), 0);
				// kd and ka are modulated by the texture
				radiance += ka * lights[i].La +
                           (kd * texColor * cost + material.ks * pow(cosd, material.shininess)) * lights[i].Le;
			}
			fragmentColor = vec4(radiance, 1);
		}
	)";
public:
    TerrainShader() { create(vertexSource, fragmentSource, "fragmentColor"); }

    void Bind(RenderState state) {
        Use(); 		// make this program run
        setUniform(state.MVP, "MVP");
        setUniform(state.M, "M");
        setUniform(state.Minv, "Minv");
        setUniform(state.wEye, "wEye");
        setUniformMaterial(*state.material, "material");

        setUniform((int)state.lights.size(), "nLights");
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
        for (unsigned int i = 0; i < nStrips; i++) glDrawArrays(GL_TRIANGLE_STRIP, i *  nVtxPerStrip, nVtxPerStrip);
    }
};

class Terrain : public ParamSurface {
public:
    Terrain() { create(); }

    VertexData genVertexData(float u, float v) {
        VertexData vd;
        vd.position = {u, v, h(u, v)};
        vd.normal = normal(u, v);
        return vd;
    }

    float h(float u, float v) {
        u *= 10;
        v *= 10;
        float h = cosf(u-2-0.3*(v-2))+cosf(v-2+0.3*(u-2))- 0.1 *(abs(v-2)- abs(u-2));
        return h;
    }

    vec3 normal(float u, float v) {
        u *= 10;
        v *= 10;
        vec3 normal;
        normal.x =  -1 * (-sinf(u - 3 / 10 * (v - 2) - 2)
                - (u - 2) / 10 / abs(u -2)
                - 3 / 10 * sinf(3 / 10 * (u - 2) + v - 2));
        normal.y = -1 * (-sinf(v - 3 / 10 * (u - 2) - 2)
                   - (v - 2) / 10 / abs(v -2)
                   - 3 / 10 * sinf(3 / 10 * (v - 2) + u - 2));
        normal.z = 1;
        return normalize(normal);
    }
};

class Route : public ParamSurface {
    std::vector<vec2> cps;

    Terrain & terrain;

    vec2 v(const vec2 & p1, float t1, const vec2 & p2, float t2, const vec2 & p3, float t3) const {
        return ((p3 - p2) * (1 / (t3 - t2)) + ((p2 - p1) * (1 / (t2 - t1)))) * 0.5f;
    }

    int posCalculator(int pos) const {
        while (pos < 0)
            pos += cps.size();
        while (pos >= cps.size())
            pos -= cps.size();
        return pos;
    }

    vec2 rHelper(
            const vec2 & p1, float t1, const vec2 & p2, float t2, const vec2 & p3, float t3, const vec2 & p4, float t4,
            float t
    ) const {
        /*if (t1 > t2) {
            t += 3600;
            t2 += 3600;
            t4 += 3600;
            t3 += 3600;
        }
        if (t < t2)
            t += 3600;
        if (t3 < t2)
            t3 += 3600;
        if (t4 < t2)
            t4 += 3600;*/

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
        /*if (t1 > t2) {
            t += 3600;
            t2 += 3600;
            t4 += 3600;
            t3 += 3600;
        }
        if (t < t2)
            t += 3600;
        if (t3 < t2)
            t3 += 3600;
        if (t4 < t2)
            t4 += 3600;*/

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

    vec2 r(float t) const {
        if (cps.size() < 2)
            return {0, 0};
        int pos = t * (float)(cps.size() - 1);
        return rHelper(
                cps[posCalculator(pos - 1)],
                (pos - 1) / (cps.size() - 1),
                cps[posCalculator(pos)],
                (pos) / (cps.size() - 1),
                cps[posCalculator(pos + 1)],
                (pos + 1) / (cps.size() - 1),
                cps[posCalculator(pos + 2)],
                (pos + 2) / (cps.size() - 1),
                t
                );
    }

    vec2 rDerivative(float t) const {
        if (cps.size() < 2)
            return {0, 0};
        // t = [0, 1]
        // 0 = 0, 1 = cps.size() - 1
        int pos = t * (float)(cps.size() - 1);
        return rDerivativeHelper(
                cps[posCalculator(pos - 1)],
                (pos - 1) / (cps.size() - 1),
                cps[posCalculator(pos)],
                (pos) / (cps.size() - 1),
                cps[posCalculator(pos + 1)],
                (pos + 1) / (cps.size() - 1),
                cps[posCalculator(pos + 2)],
                (pos + 2) / (cps.size() - 1),
                t
        );
    }

    const float h0 = 0.05f;

public:
    vec3 s(float t) {
        vec3 position;
        vec2 position2d = r(t);
        position.x = position2d.x;
        position.y = position2d.y;
        position.z = terrain.h(position.x, position.y) + h0;
        return position;;
    }

    vec3 tangent(float t) {
        vec2 positionDerivative = rDerivative(t);
        vec3 tangentVector;
        vec2 position = r(t);
        tangentVector.x = positionDerivative.x;
        tangentVector.y = positionDerivative.y;
        vec3 terrainNormal = terrain.normal(position.x, position.y);
        tangentVector.y = terrainNormal.x * positionDerivative.x + terrainNormal.y * positionDerivative.y;
        return normalize(tangentVector);
    }

    vec3 normal(float t) {
        vec2 position = r(t);
        return normalize(terrain.normal(position.x, position.y));
    }

    vec3 biNormal(float t) {
        return normalize(cross(normal(t), tangent(t)));
    }

    float X(float v) {
        return sinf(v * 2 * M_PI) * h0;
    }

    float Y(float v) {
        return cosf(v * 2 * M_PI) * h0;
    }

    float XDerivative(float v) {
        return 2 * M_PI * cosf(v * 2 * M_PI) * h0;
    }

    float YDerivative(float v) {
        return -2 * M_PI * sinf(v * 2 * M_PI) * h0;
    }

public:
    vec3 p(float u, float v) {
        vec3 position;
        position = biNormal(u) * X(v) + normal(u) * Y(v);
        return position;
    }

    Route(Terrain & _terrain) :
        terrain{_terrain} { create(10, 250); }

    void addControlPoint(vec2 cp) {
        cps.push_back(cp);
    }

    VertexData genVertexData(float u, float v) {
        VertexData vd;
        vd.position = s(u) + p(u, v);
        vd.normal = normalize(biNormal(u) * YDerivative(v) - normal(u) * XDerivative(v));
        return vd;
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

    virtual void Animate(float tstart, float tend) {}
};

struct Camera
{
    vec3 wEye, wLookat, wVup;
    float fov, asp, fp, bp;
    Route & route;
public:
    Camera(Route & _route) : route{_route} {
        asp = (float) windowWidth / windowHeight;
        fov = 75.0f * (float) M_PI / 180.0f;
        fp = 1;
        bp = 10;
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
        const int den = 100;
        const float l = 0.05;
        t = fmod(t, den) / den;
        wEye = route.s(t) + route.normal(t) * l;
        wLookat = wEye + route.tangent(t);
        wVup = route.normal(t);
    }
};


class Scene {
    std::vector<Object *> objects;
    Camera * camera;
    bool animation = false;
    std::vector<Light> lights;
public:
    void build() {
        Material * material0 = new Material;
        material0->kd = vec3(0.6f, 0.4f, 0.2f);
        material0->ks = vec3(4, 4, 4);
        material0->ka = vec3(0.1f, 0.1f, 0.1f);
        material0->shininess = 100;

        Material * material1 = new Material;
        material1->kd = vec3(0.8f, 0.6f, 0.4f);
        material1->ks = vec3(0.3f, 0.3f, 0.3f);
        material1->ka = vec3(0.2f, 0.2f, 0.2f);
        material1->shininess = 30;

        camera->wEye = vec3(0, 0, 6);
        camera->wLookat = vec3(0, 0, 0);
        camera->wVup = vec3(0, 1, 0);

        lights.resize(2);
        lights[0].wLightPos = vec4(5, 5, 4, 0);
        lights[0].La = vec3(0.1f, 0.1f, 1);
        lights[0].Le = vec3(3, 0, 0);

        lights[1].wLightPos = vec4(5, 10, 20, 0);
        lights[1].La = vec3(0.2f, 0.2f, 0.2f);
        lights[1].Le = vec3(0, 3, 0);
    }

    ~Scene() {
        delete camera;
        for (int i = 0; i < objects.size(); ++i)
            delete objects[i];
    }

    void Render() {
        RenderState state;
        state.wEye = camera->wEye;
        state.V = camera->V();
        state.P = camera->P();
        state.lights = lights;
        for (Object * obj : objects)
            obj->draw(state);
    }

    void startAnimation() {
        animation = true;
    }

    void animate(float tstart, float tend) {
        if (animation)
            camera->animate(tend);
        for (unsigned int i = 0; i < lights.size(); i++) {
            lights[i].animate(tend); }
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
    glClearColor(	0.529, 0.808, 0.98, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    scene.Render();
    glutSwapBuffers();
}

void onKeyboard(unsigned char key, int pX, int pY) {

}

void onKeyboardUp(unsigned char key, int pX, int pY) {
}

void onMouseMotion(
        int pX, int pY
) {
}

void onMouse(
        int button, int state, int pX, int pY
) {

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
