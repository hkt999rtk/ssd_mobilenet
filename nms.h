#pragma once

class BoundingBox {
    public:
        int minX;
        int minY;
        int maxX;
        int maxY;
        int score;
        int classId;

    public:
        BoundingBox() {} // default constructor
        BoundingBox(int minX, int minY, int maxX, int maxY, int score, int classId);
        BoundingBox(const BoundingBox &c) { minX = c.minX; minY = c.minY; maxX = c.maxX; maxY = c.maxY; score = c.score; classId = c.classId; }
        ~BoundingBox() {}
        BoundingBox &operator= (BoundingBox &box);
        BoundingBox &operator*= (BoundingBox &box);
        BoundingBox &operator+= (BoundingBox &box);
        bool operator<(BoundingBox &box);

        inline int GetClassId() { return classId; }

        int Area();
        void Delete();
        int IsDeleted();
        void Print();
        int IoU(BoundingBox &input); /* 100% ratio */
};

#define MAX_BOXES       (20)
#define MAX_CLASSES     (90)
#define IOU_THRESHOLD   (50)

class NmsCb {
    public:
        NmsCb() {}
        virtual ~NmsCb() {}

    public:
        virtual int callback(BoundingBox &boundingBox) = 0;
};

class ImageClass {
    protected:
        BoundingBox boxArray[MAX_BOXES];
        BoundingBox pickArray[MAX_BOXES];
        int numBox;
        int numPicked;
        int classId;

    public:
        ImageClass() { numBox = 0; numPicked = 0; classId = -1; }
        ~ImageClass() {}
        int AddBoundingBox( BoundingBox &box );

        inline int GetClassId() { return classId; }
        inline void SetClassId(int id) { classId = id; }

        void Dump(NmsCb &cb);

    public:
        void SortBoxes();
        void Go(int overlayThreshold);

    friend class NmsPostPorcess;
};

class NmsPostProcess {
    protected:
        ImageClass imageClass[MAX_CLASSES];
        int numClasses;

    public:
        NmsPostProcess() { numClasses = 0; }
        ~NmsPostProcess() {}

    public:
        int AddBoundingBox( BoundingBox &box );
        void Go(int overlayThreshold, NmsCb &cb);
};

