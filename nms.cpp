#include "nms.h"
#include <cstdio>
#include <vector>
#include <algorithm>

#define MIN(a,b) ((a<b)?a:b)
#define MAX(a,b) ((a<b)?b:a)

using namespace std;
BoundingBox::BoundingBox(int minX, int minY, int maxX, int maxY, int score, int classId):
    minX(minX), minY(minY), maxX(maxX), maxY(maxY), score(score), classId(classId)
{
}

BoundingBox &BoundingBox::operator=(BoundingBox &input)
{
    minX = input.minX;
    minY = input.minY;
    maxX = input.maxX;
    maxY = input.maxY;
    score = input.score;
    classId = input.classId;

    return *this;
}

BoundingBox &BoundingBox::operator*=(BoundingBox &input)
{
    minX = MAX(minX, input.minX);
    minY = MAX(minY, input.minY);
    maxX = MIN(maxX, input.maxX);
    maxY = MIN(maxY, input.maxY);

    return *this;
}

BoundingBox &BoundingBox::operator+=(BoundingBox &input)
{
    minX = MIN(minX, input.minX);
    minY = MIN(minY, input.minY);
    maxX = MAX(maxX, input.maxX);
    maxY = MAX(maxY, input.maxY);

    return *this;
}

int BoundingBox::Area()
{
    return (maxX - minX + 1 ) * (maxY - minY + 1 );
}

void BoundingBox::Delete()
{
    score = maxX = minX = maxY = minY = 0;
}

int BoundingBox::IsDeleted()
{
    return (score == 0 && maxX == 0 && minX == 0 && maxY == 0 && minY == 0);
}

void BoundingBox::Print()
{
    printf("(%d,%d)-(%d,%d)\r\n", minX, minY, maxX, maxY);
}

bool BoundingBox::IsOverlap(BoundingBox &dst)
{
    BoundingBox src(*this);
    if ((src.maxX - dst.minX > 0 && src.maxY - dst.minY > 0) &&
        (dst.maxX - src.minX > 0 && dst.maxY - src.minY > 0)) {
        return true;
    }
    else {
        return false;
    }
}
int BoundingBox::IoU(BoundingBox &input)
{
    if (!IsOverlap(input))
        return 0;
    BoundingBox inter(*this);
    inter *= input;
    int interArea = inter.Area();

    return (interArea * 100) / (Area() + input.Area() - interArea); /* IoU */
}

int ImageClass::AddBoundingBox( BoundingBox &box )
{
    boxArray.push_back(box);
    return 0;
}

void ImageClass::SortBoxes()
{
    //sort(boxArray.begin(), boxArray.end());
#if 1
    // bubble sort
    int numBox = boxArray.size();
    for (int i=0; i<numBox-1; i++) {
        for (int j=i+1; j<numBox; j++) {
            if (boxArray[i] < boxArray[j]) {
                BoundingBox t = boxArray[i];
                boxArray[i] = boxArray[j];
                boxArray[j] = t;
            }
        }
    }
#endif
}

void ImageClass::Go(int overlayThreshold)
{
    SortBoxes();

    int pivot = 0, numBox = boxArray.size();
    while (pivot < numBox) {
        if (boxArray[pivot].IsDeleted()) {
            pivot++;
            continue;
        }
        BoundingBox pickedBox = boxArray[pivot++];
        int candidate = pivot;
        while (candidate < numBox) {
            if (boxArray[candidate].IsDeleted()) {
                candidate++;
                continue;
            }
            int iou = pickedBox.IoU( boxArray[candidate] );
            if (iou > overlayThreshold) {
                boxArray[candidate].Delete();
            }
            candidate++;
        }
        pickArray.push_back(pickedBox);
    }
}

void ImageClass::Dump(NmsCb &cb)
{
    int numPicked = pickArray.size();
    for (int i=0; i<numPicked; i++) {
        BoundingBox b = pickArray[i];
        cb.callback(b);
    }
}

int NmsPostProcess::AddBoundingBox( BoundingBox &box )
{
    // TODO: rewrite this with map structure to eliminitate the slow linear search
    for (int i=0; i<imageClass.size(); i++) {
        if (imageClass[i].GetClassId() == box.GetClassId()) {
            imageClass[i].AddBoundingBox(box);
            return 0;
        }
    }

    ImageClass ic;
    ic.SetClassId( box.GetClassId() );
    ic.AddBoundingBox( box );

    imageClass.push_back(ic);

    return 0;
}

void NmsPostProcess::Go(int overlayThreshold, NmsCb &cb)
{
    for (int i=0; i<imageClass.size(); i++) {
        imageClass[i].Go(overlayThreshold);
        imageClass[i].Dump(cb);
    }
}
