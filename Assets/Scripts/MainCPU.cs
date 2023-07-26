using UnityEngine;
using UnityEngine.UI;
using Unity.Jobs;
using Unity.Collections;
using Unity.Burst;
using Unity.Mathematics;

struct Boid
{
  public float2 pos;
  public float2 vel;
}

public class MainCPU : MonoBehaviour
{
    [SerializeField] 
    public int numBoids = 200;
    int jobLimit = 1 << 18;
    
    [SerializeField] 
    float maxSpeed = 2;
    
    [SerializeField] 
    float edgeMargin = .5f;
    
    [SerializeField] 
    float visualRange = .5f;
    float visualRangeSq => visualRange * visualRange;
   
    [SerializeField] 
    float minDistance = 0.15f;
    float minDistanceSq => minDistance * minDistance;
    
    [SerializeField] 
    float cohesionFactor = 2;
    
    [SerializeField] 
    float separationFactor = 1;
   
    [SerializeField] 
    float alignmentFactor = 5;

    [SerializeField] Material boidMat;
     Vector2[] triangleVerts;
    GraphicsBuffer trianglePositions;

    float minSpeed;
    float turnSpeed;

    NativeArray<Boid> boids;
    NativeArray<Boid> boidsTemp;

    float xBound, yBound;
    RenderParams rp;
    ComputeBuffer boidBuffer;
    ComputeBuffer boidBufferOut;

    // Index is particle ID, x value is position flattened to 1D array, y value is grid cell offset
    NativeArray<int2> grid;
    NativeArray<int> gridOffsets;
    int gridDimY, gridDimX, gridTotalCells;
    float gridCellSize;

    void Awake()
    {
        triangleVerts = getTriangleVerts();
    }

    Vector2[] getTriangleVerts()
    {
        return new Vector2[] {
        new Vector2(-.4f, -.5f),
        new Vector2(0, .5f),
        new Vector2(.4f, -.5f),
        };
    }
    
    // Start is called before the first frame update
    void Start()
    {

        xBound = Camera.main.orthographicSize * Camera.main.aspect - edgeMargin;
        yBound = Camera.main.orthographicSize - edgeMargin;
        turnSpeed = maxSpeed * 3;
        minSpeed = maxSpeed * 0.75f;
        
        boidBuffer = new ComputeBuffer(numBoids, 16);
        boidBufferOut = new ComputeBuffer(numBoids, 16);

        // Populate initial boids
        boids = new NativeArray<Boid>(numBoids, Allocator.Persistent);
        boidsTemp = new NativeArray<Boid>(numBoids, Allocator.Persistent);
        for (int i = 0; i < numBoids; i++)
        {
            var pos = new float2(UnityEngine.Random.Range(-xBound, xBound), UnityEngine.Random.Range(-yBound, yBound));
            var vel = new float2(UnityEngine.Random.Range(-maxSpeed, maxSpeed), UnityEngine.Random.Range(-maxSpeed, maxSpeed));
            var boid = new Boid();
            boid.pos = pos;
            boid.vel = vel;
            boids[i] = boid;
        }
        
        boidBuffer.SetData(boids);

        // Set render params
        rp = new RenderParams(boidMat);
        rp.matProps = new MaterialPropertyBlock();
        rp.matProps.SetBuffer("boids", boidBuffer);
        rp.worldBounds = new Bounds(Vector3.zero, Vector3.one * 3000);
        trianglePositions = new GraphicsBuffer(GraphicsBuffer.Target.Structured, 3, 8);
        trianglePositions.SetData(triangleVerts);
        rp.matProps.SetBuffer("_Positions", trianglePositions);

        // Spatial grid setup
        gridCellSize = visualRange;
        gridDimX = Mathf.FloorToInt(xBound * 2 / gridCellSize) + 30;
        gridDimY = Mathf.FloorToInt(yBound * 2 / gridCellSize) + 30;
        gridTotalCells = gridDimX * gridDimY;
      
        //Debug.Log("grid total cells = "+gridTotalCells);
        //Debug.Log(numBoids+" numBoids "+jobLimit+ "jobLimit");
        
        grid = new NativeArray<int2>(numBoids, Allocator.Persistent);
        gridOffsets = new NativeArray<int>(gridTotalCells, Allocator.Persistent);
    }

    // Update is called once per frame
    void Update()
    {
        ClearGrid();
        UpdateGrid();
        GenerateGridOffsets();
        RearrangeBoids();

        for (int i = 0; i < numBoids; i++)
        {
          var boid = boidsTemp[i];
          MergedBehaviours(ref boid);
          LimitSpeed(ref boid);
          KeepInBounds(ref boid);

          // Update boid position
          boid.pos += boid.vel * Time.deltaTime;
          boids[i] = boid;
        }

        // Send data to gpu buffer
        boidBuffer.SetData(boids);

        // Actually draw the boids
        Graphics.RenderPrimitives(rp, MeshTopology.Triangles, numBoids * 3);
    }

    void ClearGrid()
    {
        for (int i = 0; i < gridTotalCells; i++)
        {
            gridOffsets[i] = 0;
        }
    }

    void UpdateGrid()
    {
        for (int i = 0; i < numBoids; i++)
        {
            int id = getGridID(boids[i]);
            var boidGrid = grid[i];
            boidGrid.x = id;
            boidGrid.y = gridOffsets[id];
            grid[i] = boidGrid;
            gridOffsets[id]++;
        }
    }

    void GenerateGridOffsets()
    {
        for (int i = 1; i < gridTotalCells; i++)
        {
            gridOffsets[i] += gridOffsets[i - 1];
        }
    }

    void RearrangeBoids()
    {
        for (int i = 0; i < numBoids; i++)
        {
            int gridID = grid[i].x;
            int cellOffset = grid[i].y;
            int index = gridOffsets[gridID] - 1 - cellOffset;
            boidsTemp[index] = boids[i];
        }
    }

    int getGridID(Boid boid)
    {
        int gridX = Mathf.FloorToInt(boid.pos.x / gridCellSize + gridDimX / 2);
        int gridY = Mathf.FloorToInt(boid.pos.y / gridCellSize + gridDimY / 2);
        return (gridDimX * gridY) + gridX;
    }

    void MergedBehaviours(ref Boid boid)
    {
        float2 center = float2.zero;
        float2 close = float2.zero;
        float2 avgVel = float2.zero;
        int neighbours = 0;

        var gridXY = getGridLocation(boid);
        int gridCell = getGridIDbyLoc(gridXY);

        for (int y = gridCell - gridDimX; y <= gridCell + gridDimX; y += gridDimX)
        {
            int start = gridOffsets[y - 2];
            int end = gridOffsets[y + 1];
            for (int i = start; i < end; i++)
            {
                Boid other = boidsTemp[i];
                var diff = boid.pos - other.pos;
                var distanceSq = math.dot(diff, diff);
                if (distanceSq > 0 && distanceSq < visualRangeSq)
                {
                    if (distanceSq < minDistanceSq)
                    {
                        close += diff / distanceSq;
                    }
                    center += other.pos;
                    avgVel += other.vel;
                    neighbours++;
                }
            }
        }
        if (neighbours > 0)
        {
            center /= neighbours;
            avgVel /= neighbours;

            boid.vel += (center - boid.pos) * (cohesionFactor * Time.deltaTime);
            boid.vel += (avgVel - boid.vel) * (alignmentFactor * Time.deltaTime);
        }

        boid.vel += close * (separationFactor * Time.deltaTime);
    }

    void LimitSpeed(ref Boid boid)
    {
        var speed = math.length(boid.vel);
        var clampedSpeed = Mathf.Clamp(speed, minSpeed, maxSpeed);
        boid.vel *= clampedSpeed / speed;
    }

    // Keep boids on screen, turn aroun if out of bounds
    void KeepInBounds(ref Boid boid)
    {
        if (Mathf.Abs(boid.pos.x) > xBound)
        {
            boid.vel.x -= Mathf.Sign(boid.pos.x) * Time.deltaTime * turnSpeed;
        }
        if (Mathf.Abs(boid.pos.y) > yBound)
        {
            boid.vel.y -= Mathf.Sign(boid.pos.y) * Time.deltaTime * turnSpeed;
        }
    }

    int getGridIDbyLoc(int2 cell)
    {
        return (gridDimX * cell.y) + cell.x;
    }

    int2 getGridLocation(Boid boid)
    {
        int gridX = Mathf.FloorToInt(boid.pos.x / gridCellSize + gridDimX / 2);
        int gridY = Mathf.FloorToInt(boid.pos.y / gridCellSize + gridDimY / 2);
        return new int2(gridX, gridY);
    }
}
