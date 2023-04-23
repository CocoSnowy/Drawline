using UnityEngine;

public class pictures : MonoBehaviour
{
    public Texture2D image; // Ҫʹ�õ�ͼƬ
    public float spacing = 0.0001f; // ͼƬ֮��ļ��
    public int numImages = 100; // ͼƬ����
    public Vector3 startPosition; // ��ʼλ��
    public Vector3 endPosition; // ����λ��

    void Start()
    {
        // ����һ����ͼƬ
        for (int i = 0; i < numImages; i++)
        {
            // ����ƽ�����
            GameObject plane = GameObject.CreatePrimitive(PrimitiveType.Plane);
            plane.transform.SetParent(transform);

            // ��������
            //plane.GetComponent<Renderer>().material.mainTexture = image;
            Renderer renderer = plane.GetComponent<Renderer>();
            Material material = renderer.material;
            material.mainTexture = image;
            Shader shader = Shader.Find("Mobile/Particles/Alpha Blended");
            material.shader = shader;


            // ����λ��
            float t = (float)i / (numImages - 1);
            plane.transform.position = Vector3.Lerp(startPosition, endPosition, t) + Vector3.forward * i * spacing;

            // ������ת������
            plane.transform.rotation = Quaternion.identity;
            plane.transform.localScale = Vector3.one * 0.1f; // ������С
        }
    }
}