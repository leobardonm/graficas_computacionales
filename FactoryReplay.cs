using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.Networking;
[Serializable] public class DropZoneDTO { public int x; public int y; }
[Serializable] public class AgentDTO { public int id; public int x; public int y; public bool carrying; public bool available; }
[Serializable] public class BoxDTO { public int id; public int x; public int y; }
[Serializable] public class FrameDTO { public List<AgentDTO> agents; public List<BoxDTO> boxes; public int delivered; }
[Serializable] public class ReplayDTO {
    public int gridWidth;
    public int gridHeight;
    public DropZoneDTO dropZone;
    public List<FrameDTO> frames;
}

public class FactoryReplay : MonoBehaviour
{
    [Header("Replay JSON (en StreamingAssets)")]
    public string fileName = "factory_trace.json";

    [Header("Prefabs (opcionales)")]
    public GameObject agentPrefab;    // Sphere o tu agente
    public GameObject boxPrefab;      // Cube o tu caja
    public GameObject dropZonePrefab; // Plane o marcador

    [Header("Animación")]
    public float cellSize = 1.0f;   // tamaño de celda
    public float stepTime = 0.25f;  // segundos por frame
    public bool tween = true;       // interpolar movimiento
    public bool loop = false;       // repetir al terminar

    private ReplayDTO replay;
    private Dictionary<int, GameObject> agentGOs = new Dictionary<int, GameObject>();
    private Dictionary<int, GameObject> boxGOs = new Dictionary<int, GameObject>();
    private GameObject dropGO;
    private Coroutine playRoutine;

    IEnumerator Start()
    {
        yield return LoadReplayAsync();
        if (replay == null || replay.frames == null || replay.frames.Count == 0)
        {
            Debug.LogError("Replay vacío o inválido.");
            yield break;
        }
        BuildScene();
        playRoutine = StartCoroutine(Play());
    }

    // Carga robusta desde StreamingAssets (Editor/PC/Android/WebGL)
    IEnumerator LoadReplayAsync()
    {
        string path = Path.Combine(Application.streamingAssetsPath, fileName);

#if UNITY_EDITOR || UNITY_STANDALONE || UNITY_IOS
        // En estas plataformas podemos leer directo si no es WebGL
        if (path.StartsWith("file://") == false)
            path = "file://" + path;
#endif
        using (UnityWebRequest req = UnityWebRequest.Get(path))
        {
            yield return req.SendWebRequest();
#if UNITY_2020_3_OR_NEWER
            if (req.result != UnityWebRequest.Result.Success)
#else
            if (req.isNetworkError || req.isHttpError)
#endif
            {
                Debug.LogError("Error al leer JSON: " + req.error + "\nPath: " + path);
                yield break;
            }
            string json = req.downloadHandler.text;
            replay = JsonUtility.FromJson<ReplayDTO>(json);
        }
    }

    void BuildScene()
    {
        // (Opcional) dibujar grid con Debug.DrawLine (solo Editor/Play)
        for (int x = 0; x < replay.gridWidth; x++)
        {
            Debug.DrawLine(ToWorld(x, 0), ToWorld(x, replay.gridHeight - 1), Color.gray, 999f);
        }
        for (int y = 0; y < replay.gridHeight; y++)
        {
            Debug.DrawLine(ToWorld(0, y), ToWorld(replay.gridWidth - 1, y), Color.gray, 999f);
        }

        // Plano del tamaño del grid
        Vector3 gridCenter = new Vector3(
            (replay.gridWidth * cellSize) / 2f,
            0f,
            (replay.gridHeight * cellSize) / 2f
        );
        GameObject gridPlane;
        if (dropZonePrefab != null)
        {
            gridPlane = Instantiate(dropZonePrefab, gridCenter, Quaternion.identity);
        }
        else
        {
            gridPlane = GameObject.CreatePrimitive(PrimitiveType.Plane);
            gridPlane.transform.position = gridCenter;
        }
        gridPlane.name = "GridPlane";
        // El Plane de Unity es 10x10 unidades, así que hay que escalarlo
        float extraScale = 1.15f; // 10% más grande
        gridPlane.transform.localScale = new Vector3(
            (replay.gridWidth * cellSize * extraScale) / 10f,
            1f,
            (replay.gridHeight * cellSize * extraScale) / 10f
        );

        // DropZone
        if (dropZonePrefab != null)
        {
            dropGO = Instantiate(dropZonePrefab, ToWorld(replay.dropZone.x, replay.dropZone.y), Quaternion.identity);
            dropGO.name = "DropZone";
            dropGO.transform.localScale = Vector3.one * (cellSize * 0.95f);
        }

        // Frame 0: crear agentes/cajas
        var f0 = replay.frames[0];

        foreach (var a in f0.agents)
        {
            GameObject go = agentPrefab != null ? Instantiate(agentPrefab)
                                                : GameObject.CreatePrimitive(PrimitiveType.Sphere);
            go.transform.position = ToWorld(a.x, a.y);
            go.transform.localScale = Vector3.one * (cellSize * 0.8f);
            go.name = $"Agent_{a.id}";
            agentGOs[a.id] = go;
        }

        foreach (var b in f0.boxes)
        {
            GameObject go = boxPrefab != null ? Instantiate(boxPrefab)
                                              : GameObject.CreatePrimitive(PrimitiveType.Cube);
            go.transform.position = ToWorld(b.x, b.y);
            go.transform.localScale = Vector3.one * (cellSize * 0.8f);
            go.name = $"Box_{b.id}";
            boxGOs[b.id] = go;
        }
    }

    IEnumerator Play()
    {
        do
        {
            for (int i = 1; i < replay.frames.Count; i++)
            {
                var fr = replay.frames[i];

                // Actualiza/crea cajas vivas
                HashSet<int> alive = new HashSet<int>();
                foreach (var b in fr.boxes)
                {
                    alive.Add(b.id);
                    if (!boxGOs.ContainsKey(b.id))
                    {
                        GameObject go = boxPrefab != null ? Instantiate(boxPrefab)
                                                          : GameObject.CreatePrimitive(PrimitiveType.Cube);
                        go.transform.localScale = Vector3.one * (cellSize * 0.8f);
                        go.name = $"Box_{b.id}";
                        boxGOs[b.id] = go;
                    }
                    MoveGO(boxGOs[b.id], ToWorld(b.x, b.y));
                }
                // Destruir cajas que ya no están (entregadas)
                var toRemove = new List<int>();
                foreach (var kv in boxGOs)
                    if (!alive.Contains(kv.Key)) toRemove.Add(kv.Key);
                foreach (int id in toRemove)
                {
                    Destroy(boxGOs[id]);
                    boxGOs.Remove(id);
                }

                // Actualizar/crear agentes
                foreach (var a in fr.agents)
                {
                    if (!agentGOs.ContainsKey(a.id))
                    {
                        GameObject go = agentPrefab != null ? Instantiate(agentPrefab)
                                                            : GameObject.CreatePrimitive(PrimitiveType.Sphere);
                        go.transform.localScale = Vector3.one * (cellSize * 0.8f);
                        go.name = $"Agent_{a.id}";
                        agentGOs[a.id] = go;
                    }
                    MoveGO(agentGOs[a.id], ToWorld(a.x, a.y));

                    // (Opcional) Cambiar tamaño/indicador si va cargando
                    // agentGOs[a.id].transform.localScale = Vector3.one * (cellSize * (a.carrying ? 1.0f : 0.8f));
                }

                yield return new WaitForSeconds(stepTime);
            }
        } while (loop);
    }

    void MoveGO(GameObject go, Vector3 target)
    {
        if (!tween)
        {
            go.transform.position = target;
        }
        else
        {
            // Tween simple por frame (si quieres tweens por-agente, usa una corrutina por id)
            StartCoroutine(TweenMove(go, target, stepTime));
        }
    }

    IEnumerator TweenMove(GameObject go, Vector3 target, float dur)
    {
        Vector3 start = go.transform.position;
        float t = 0f;
        while (t < 1f)
        {
            t += Time.deltaTime / Mathf.Max(dur, 0.0001f);
            go.transform.position = Vector3.Lerp(start, target, Mathf.Clamp01(t));
            yield return null;
        }
        go.transform.position = target;
    }

    Vector3 ToWorld(int gx, int gy)
    {
        // Mapea celda (gx,gy) a mundo (X,Z), Y=0
        return new Vector3(gx * cellSize, 0f, gy * cellSize);
    }

    // ---- Controles simples (opcional) ----
    public void Restart()
    {
        if (playRoutine != null) StopCoroutine(playRoutine);
        foreach (var go in agentGOs.Values) Destroy(go);
        foreach (var go in boxGOs.Values) Destroy(go);
        agentGOs.Clear(); boxGOs.Clear();
        BuildScene();
        playRoutine = StartCoroutine(Play());
    }
}