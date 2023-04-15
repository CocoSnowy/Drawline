using UnityEngine;

public class Picture_2_Face : MonoBehaviour
{
    public Texture2D image; // 用于转换的图片
    public float cubeScale = 1.0f; // 面片的缩放比例
    public int faceIndex; // 立方体上要附加纹理的面的索引
    public int thick;//厚度

    void Start()
    {
        // 创建一个新的平面对象
        GameObject Cube = GameObject.CreatePrimitive(PrimitiveType.Cube);
        Cube.transform.localScale = new Vector3(image.width * cubeScale, image.height * cubeScale, thick);

        // 获取cube的网格渲染器
        MeshRenderer renderer = Cube.GetComponent<MeshRenderer>();

        // 获取立方体的材质，并将纹理附加到指定的面
        Material material = renderer.materials[faceIndex];
        material.mainTexture = image;
        Shader shader = Shader.Find("Mobile/Particles/Alpha Blended");
        material.shader = shader;
    }
}





    
