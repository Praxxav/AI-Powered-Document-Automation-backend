-- CreateTable
CREATE TABLE "Document" (
    "id" TEXT NOT NULL,
    "status" TEXT NOT NULL,
    "insights" JSONB,
    "fullText" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "Document_pkey" PRIMARY KEY ("id")
);
