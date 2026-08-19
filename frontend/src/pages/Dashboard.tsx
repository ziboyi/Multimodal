import { useState } from 'react'
import { Link } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Plus, BookOpen, FileText, Trash2, Loader2 } from 'lucide-react'
import apiClient from '../api/client'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Dialog } from '@/components/ui/dialog'
import { Badge } from '@/components/ui/badge'
import { Skeleton } from '@/components/ui/skeleton'

interface KnowledgeBase {
  id: string
  name: string
  description: string | null
  document_count: number
  created_at: string
}

export default function Dashboard() {
  const [showCreate, setShowCreate] = useState(false)
  const [newName, setNewName] = useState('')
  const [newDesc, setNewDesc] = useState('')
  const queryClient = useQueryClient()

  const { data: kbs, isLoading } = useQuery({
    queryKey: ['knowledge-bases'],
    queryFn: async () => {
      const res = await apiClient.get('/kb')
      return res.data as KnowledgeBase[]
    },
  })

  const createMutation = useMutation({
    mutationFn: async () => {
      await apiClient.post('/kb', { name: newName, description: newDesc || null })
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['knowledge-bases'] })
      setShowCreate(false)
      setNewName('')
      setNewDesc('')
    },
  })

  const [showDelete, setShowDelete] = useState<string | null>(null)

  const deleteMutation = useMutation({
    mutationFn: async (kbId: string) => {
      await apiClient.delete(`/kb/${kbId}`)
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['knowledge-bases'] })
      setShowDelete(null)
    },
  })

  const confirmDelete = (kbId: string) => {
    setShowDelete(kbId)
  }

  const handleDelete = () => {
    if (showDelete) {
      deleteMutation.mutate(showDelete)
    }
  }

  return (
    <div className="p-8">
      {/* 头部 */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">我的知识库</h1>
          <p className="text-sm text-gray-500 mt-1">管理和组织您的文档知识库</p>
        </div>
        <Button onClick={() => setShowCreate(true)}>
          <Plus size={16} className="mr-2" /> 新建知识库
        </Button>
      </div>

      {/* 知识库网格 */}
      {isLoading ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {[1, 2, 3].map((i) => (
            <Card key={i}>
              <CardHeader>
                <Skeleton className="h-5 w-3/4 mb-2" />
                <Skeleton className="h-4 w-full" />
              </CardHeader>
              <CardContent>
                <Skeleton className="h-4 w-1/2" />
              </CardContent>
            </Card>
          ))}
        </div>
      ) : kbs && kbs.length > 0 ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {kbs.map((kb) => (
            <div key={kb.id} className="relative group">
              <Link to={`/kb/${kb.id}`}>
                <Card className="hover:shadow-md transition-shadow cursor-pointer h-full">
                  <CardHeader>
                    <div className="flex items-start gap-3">
                      <div className="p-2 bg-blue-50 rounded-lg">
                        <BookOpen size={20} className="text-blue-600" />
                      </div>
                      <div className="flex-1 min-w-0">
                        <CardTitle className="text-base truncate">{kb.name}</CardTitle>
                        {kb.description && (
                          <CardDescription className="mt-1 line-clamp-2">
                            {kb.description}
                          </CardDescription>
                        )}
                      </div>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-1 text-xs text-gray-400">
                        <FileText size={12} />
                        <span>{kb.document_count} 个文档</span>
                      </div>
                      <span className="text-xs text-gray-400">
                        {new Date(kb.created_at).toLocaleDateString()}
                      </span>
                    </div>
                  </CardContent>
                </Card>
              </Link>
              <Button
                variant="ghost"
                size="sm"
                className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity"
                onClick={(e) => { e.preventDefault(); confirmDelete(kb.id) }}
              >
                <Trash2 size={14} className="text-red-500" />
              </Button>
            </div>
          ))}
        </div>
      ) : (
        <div className="text-center py-20">
          <BookOpen size={48} className="mx-auto text-gray-300 mb-4" />
          <p className="text-gray-500 mb-4">暂无知识库</p>
          <Button onClick={() => setShowCreate(true)} variant="outline">
            <Plus size={16} className="mr-2" /> 创建第一个知识库
          </Button>
        </div>
      )}

      {/* 创建弹窗 */}
      <Dialog open={showCreate} onClose={() => setShowCreate(false)}>
        <h2 className="text-lg font-semibold mb-4">创建知识库</h2>
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              名称
            </label>
            <Input
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
              placeholder="输入知识库名称"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              描述（可选）
            </label>
            <Textarea
              value={newDesc}
              onChange={(e) => setNewDesc(e.target.value)}
              placeholder="输入知识库描述"
              rows={3}
            />
          </div>
          <div className="flex gap-2 justify-end">
            <Button variant="outline" onClick={() => setShowCreate(false)}>
              取消
            </Button>
            <Button
              onClick={() => createMutation.mutate()}
              disabled={!newName || createMutation.isPending}
            >
              {createMutation.isPending ? '创建中...' : '创建'}
            </Button>
          </div>
        </div>
      </Dialog>

      {/* 删除确认弹窗 */}
      <Dialog open={!!showDelete} onClose={() => setShowDelete(null)}>
        <h2 className="text-lg font-semibold mb-4">确认删除</h2>
        <p className="text-sm text-gray-500 mb-6">
          确定要删除知识库 "{kbs?.find(k => k.id === showDelete)?.name}" 吗？<br/>
          所有文档、图片和索引将被永久删除，此操作不可撤销。
        </p>
        <div className="flex gap-2 justify-end">
          <Button variant="outline" onClick={() => setShowDelete(null)}>
            取消
          </Button>
          <Button
            variant="destructive"
            onClick={handleDelete}
            disabled={deleteMutation.isPending}
          >
            {deleteMutation.isPending ? <Loader2 size={14} className="animate-spin" /> : <Trash2 size={14} className="mr-1" />}
            确认删除
          </Button>
        </div>
      </Dialog>
    </div>
  )
}
