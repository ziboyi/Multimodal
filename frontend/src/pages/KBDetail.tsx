import { useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { ArrowLeft, Upload, FileText, Trash2, Loader2, File } from 'lucide-react'
import apiClient from '../api/client'
import { useDropzone } from 'react-dropzone'
import { Button } from '@/components/ui/button'
import { Card, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Skeleton } from '@/components/ui/skeleton'

interface Document {
  id: string
  filename: string
  file_type: string
  file_size: number
  status: string
  chunk_count: number
  error_message: string | null
  created_at: string
}

export default function KBDetail() {
  const { kbId } = useParams<{ kbId: string }>()
  const queryClient = useQueryClient()
  const [uploading, setUploading] = useState(false)

  const { data, isLoading } = useQuery({
    queryKey: ['documents', kbId],
    queryFn: async () => {
      const res = await apiClient.get(`/kb/${kbId}/documents`)
      return res.data as { total: number; items: Document[] }
    },
  })

  const uploadMutation = useMutation({
    mutationFn: async (files: File[]) => {
      setUploading(true)
      try {
        const formData = new FormData()
        files.forEach((f) => formData.append('files', f))
        await apiClient.post(`/kb/${kbId}/documents`, formData, {
          headers: { 'Content-Type': 'multipart/form-data' },
        })
      } finally {
        setUploading(false)
      }
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['documents', kbId] })
    },
  })

  const deleteMutation = useMutation({
    mutationFn: async (docId: string) => {
      await apiClient.delete(`/kb/${kbId}/documents/${docId}`)
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['documents', kbId] })
    },
  })

  const onDrop = (acceptedFiles: File[]) => {
    uploadMutation.mutate(acceptedFiles)
  }

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
      'application/vnd.openxmlformats-officedocument.presentationml.presentation': ['.pptx'],
      'text/markdown': ['.md'],
      'text/plain': ['.txt'],
      'image/*': ['.png', '.jpg', '.jpeg'],
    },
  })

  const statusConfig: Record<string, { label: string; variant: string }> = {
    pending: { label: '等待中', variant: 'secondary' },
    parsing: { label: '解析中', variant: 'warning' },
    chunking: { label: '分块中', variant: 'default' },
    indexing: { label: '索引中', variant: 'default' },
    completed: { label: '已完成', variant: 'success' },
    failed: { label: '失败', variant: 'destructive' },
  }

  const documents = data?.items || []

  return (
    <div className="p-8">
      {/* 头部 */}
      <div className="flex items-center gap-4 mb-8">
        <Link to="/" className="text-gray-500 hover:text-gray-900">
          <ArrowLeft size={20} />
        </Link>
        <div>
          <h1 className="text-2xl font-bold text-gray-900">知识库详情</h1>
          <p className="text-sm text-gray-500 mt-1">上传和管理文档</p>
        </div>
      </div>

      {/* 上传区域 */}
      <Card className="mb-8">
        <CardContent className="p-6">
          <div
            {...getRootProps()}
            className={`p-8 border-2 border-dashed rounded-xl text-center cursor-pointer transition ${
              isDragActive
                ? 'border-blue-500 bg-blue-50'
                : 'border-gray-200 hover:border-gray-300'
            }`}
          >
            <input {...getInputProps()} />
            <Upload size={32} className="mx-auto mb-3 text-gray-400" />
            <p className="text-gray-600">
              {isDragActive ? '拖放文件到这里...' : '拖拽文件到此处，或点击选择文件'}
            </p>
            <p className="text-sm text-gray-400 mt-1">
              支持 PDF、Word、PPT、Markdown、图片等格式
            </p>
            {uploading && (
              <div className="mt-3 flex items-center justify-center gap-2 text-blue-600">
                <Loader2 size={16} className="animate-spin" />
                <span>上传处理中...</span>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* 文档列表 */}
      <div>
        <h2 className="text-lg font-semibold mb-4">
          文档列表
          {data && <span className="text-sm font-normal text-gray-400 ml-2">({data.total})</span>}
        </h2>

        {isLoading ? (
          <div className="space-y-3">
            {[1, 2, 3].map((i) => (
              <Card key={i}>
                <CardContent className="p-4 flex items-center gap-4">
                  <Skeleton className="h-10 w-10 rounded-lg" />
                  <div className="flex-1">
                    <Skeleton className="h-4 w-1/3 mb-2" />
                    <Skeleton className="h-3 w-1/4" />
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        ) : documents.length > 0 ? (
          <div className="space-y-2">
            {documents.map((doc) => {
              const status = statusConfig[doc.status] || { label: doc.status, variant: 'secondary' }
              return (
                <Card key={doc.id}>
                  <CardContent className="p-4 flex items-center gap-4">
                    <div className="p-2 bg-gray-50 rounded-lg">
                      <FileText size={20} className="text-gray-400" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="font-medium text-gray-900 truncate">
                        {doc.filename}
                      </p>
                      <div className="flex items-center gap-3 mt-1 text-xs text-gray-400">
                        <span>{doc.file_type.toUpperCase()}</span>
                        <span>{(doc.file_size / 1024).toFixed(1)} KB</span>
                        {doc.chunk_count > 0 && <span>{doc.chunk_count} 个块</span>}
                        <span>{new Date(doc.created_at).toLocaleString()}</span>
                      </div>
                      {doc.error_message && (
                        <p className="text-xs text-red-500 mt-1">{doc.error_message}</p>
                      )}
                    </div>
                    <Badge variant={status.variant as any}>{status.label}</Badge>
                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={() => deleteMutation.mutate(doc.id)}
                    >
                      <Trash2 size={16} className="text-gray-400 hover:text-red-500" />
                    </Button>
                  </CardContent>
                </Card>
              )
            })}
          </div>
        ) : (
          <div className="text-center py-12 text-gray-400">
            <File size={40} className="mx-auto mb-3 opacity-50" />
            <p>暂无文档，上传第一个文档开始使用</p>
          </div>
        )}
      </div>
    </div>
  )
}
