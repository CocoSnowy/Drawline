using UnityEngine;
using System.IO;


public class CreatePlaneFromImage : MonoBehaviour
{
    public string imagePath;
    /*public float planeWidth = 10f;
    public float planeHeight = 10f;*/
    public float cubeScale = 1.0f; // 面片的缩放比例
    public int faceIndex; // 立方体上要附加纹理的面的索引
    public int thick;//厚度

    void Start()
    {
        // Load the texture from the PNG image
        Texture2D texture = LoadPNG(imagePath);

        // Create a new material using the texture
        Material material = new Material(Shader.Find("Standard"));
        material.mainTexture = texture;
        Shader shader = Shader.Find("Mobile/Particles/Alpha Blended");
        material.shader = shader;

        // Create a new game object and add a mesh renderer component
        //GameObject plane = new GameObject("PlaneFromImage");
        GameObject Cube = GameObject.CreatePrimitive(PrimitiveType.Cube);
        Cube.transform.localScale = new Vector3(texture.width * cubeScale, texture.height * cubeScale, thick);
        MeshRenderer renderer = Cube.AddComponent<MeshRenderer>();
        renderer.material = material;
        renderer.material.color = new Color(1f, 1f, 1f, 0.5f);
        renderer.material.SetFloat("_Alpha", 0.5f);




        // Create a mesh filter component and assign a new mesh to it
        MeshFilter filter = Cube.AddComponent<MeshFilter>();
        //filter.mesh = CreatePlaneMesh(planeWidth, planeHeight, texture.width, texture.height);
    }

    private Texture2D LoadPNG(string filePath)
    {
        Texture2D texture = null;
        byte[] fileData;

        if (File.Exists(filePath))
        {
            fileData = File.ReadAllBytes(filePath);
            texture = new Texture2D(2, 2);
            texture.LoadImage(fileData);
        }

        return texture;
    }

    private Mesh CreatePlaneMesh(float width, float height, int textureWidth, int textureHeight)
    {
        Mesh mesh = new Mesh();

        Vector3[] vertices = new Vector3[4];
        vertices[0] = new Vector3(0, 0, 0);
        vertices[1] = new Vector3(width, 0, 0);
        vertices[2] = new Vector3(0, 0, height);
        vertices[3] = new Vector3(width, 0, height);

        Vector2[] uv = new Vector2[4];
        uv[0] = new Vector2(0, 0);
        uv[1] = new Vector2(1, 0);
        uv[2] = new Vector2(0, 1);
        uv[3] = new Vector2(1, 1);

        int[] triangles = new int[6];
        triangles[0] = 0;
        triangles[1] = 2;
        triangles[2] = 1;
        triangles[3] = 2;
        triangles[4] = 3;
        triangles[5] = 1;

        mesh.vertices = vertices;
        mesh.uv = uv;
        mesh.triangles = triangles;

        return mesh;
    }
}
