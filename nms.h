#pragma once

#include <vector>
//#define MAX_CLASSES     (90)
#define IOU_THRESHOLD   (50)

using namespace std;

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

class NmsCb {
    public:
        NmsCb() {}
        virtual ~NmsCb() {}

    public:
        virtual int callback(BoundingBox &boundingBox) = 0;
};

class ImageClass {
    protected:
        vector<BoundingBox> boxArray;
        vector<BoundingBox> pickArray;
        int classId;

    public:
        ImageClass() { classId = -1; }
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
        vector<ImageClass> imageClass;

    public:
        NmsPostProcess() {}
        ~NmsPostProcess() {}

    public:
        int AddBoundingBox( BoundingBox &box );
        void Go(int overlayThreshold, NmsCb &cb);
};

