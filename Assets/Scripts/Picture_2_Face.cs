using UnityEngine;

public class Picture_2_Face : MonoBehaviour
{
    public Texture2D image; // ����ת����ͼƬ
    public float cubeScale = 1.0f; // ��Ƭ�����ű���
    public int faceIndex; // ��������Ҫ����������������
    public int thick;//���

    void Start()
    {
        // ����һ���µ�ƽ�����
        GameObject Cube = GameObject.CreatePrimitive(PrimitiveType.Cube);
        Cube.transform.localScale = new Vector3(image.width * cubeScale, image.height * cubeScale, thick);

        // ��ȡcube��������Ⱦ��
        MeshRenderer renderer = Cube.GetComponent<MeshRenderer>();

        // ��ȡ������Ĳ��ʣ����������ӵ�ָ������
        Material material = renderer.materials[faceIndex];
        material.mainTexture = image;
        Shader shader = Shader.Find("Mobile/Particles/Alpha Blended");
        material.shader = shader;
    }
}





    
