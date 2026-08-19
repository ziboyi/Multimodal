import { useState } from 'react'
import { Search as SearchIcon, FileText, Image, Sparkles, Info } from 'lucide-react'
import apiClient from '../api/client'
import { Card, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Dialog } from '@/components/ui/dialog'

interface SearchResult {
  chunk_id: string
  doc_id: string
  kb_id: string
  document_name: string
  content: string
  chunk_type: string
  page_number: number | null
  score: number
  image_url: string
  highlight: string
  images?: Array<{
    url: string
    path: string
    caption: string
  }>
  metadata?: Record<string, any>
  language?: string
}

export default function Search() {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<SearchResult[]>([])
  const [loading, setLoading] = useState(false)
  const [searched, setSearched] = useState(false)
  const [elapsedMs, setElapsedMs] = useState(0)
  const [selectedResult, setSelectedResult] = useState<SearchResult | null>(null)

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!query.trim()) return
    setLoading(true)
    const start = Date.now()
    try {
      const res = await apiClient.post('/search', {
        query,
        top_k: 10,
        search_mode: 'hybrid',
      })
      setResults(res.data.results || [])
      setElapsedMs(Date.now() - start)
      setSearched(true)
    } catch (err) {
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="p-8 max-w-4xl mx-auto">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-gray-900 flex items-center gap-2">
          <Sparkles size={24} className="text-blue-500" />
          全局检索
        </h1>
        <p className="text-sm text-gray-500 mt-1">跨知识库的多模态混合检索</p>
      </div>

      <form onSubmit={handleSearch} className="flex gap-2 mb-8">
        <div className="flex-1 relative">
          <SearchIcon size={18} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" />
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="输入检索内容，支持多语言..."
            className="w-full h-12 pl-10 pr-4 border border-gray-300 rounded-xl text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          />
        </div>
        <button
          type="submit"
          disabled={loading || !query.trim()}
          className="h-12 px-6 bg-blue-600 text-white rounded-xl font-medium hover:bg-blue-700 disabled:opacity-50 transition"
        >
          {loading ? '检索中...' : '检索'}
        </button>
      </form>

      {searched && !loading && (
        <p className="text-sm text-gray-500 mb-4">
          找到 {results.length} 个结果，耗时 {elapsedMs}ms
        </p>
      )}

      <div className="space-y-3">
        {loading && (
          <div className="text-center py-12 text-gray-400">检索中...</div>
        )}

        {!loading && searched && results.length === 0 && (
          <div className="text-center py-12 text-gray-400">
            <SearchIcon size={40} className="mx-auto mb-3 opacity-50" />
            <p>未找到相关内容</p>
          </div>
        )}

        {results.map((r, i) => (
          <Card
            key={r.chunk_id || i}
            className="hover:shadow-md transition-shadow cursor-pointer border-gray-200 hover:border-blue-300"
            onClick={() => setSelectedResult(r)}
          >
            <CardContent className="p-5">
              <div className="flex items-center gap-2 mb-2">
                <FileText size={14} className="text-gray-400" />
                <span className="text-sm font-medium text-gray-700">
                  {r.document_name}
                </span>
                {r.page_number && (
                  <span className="text-xs text-gray-400">第 {r.page_number} 页</span>
                )}
                {r.images && r.images.length > 0 && (
                  <span className="text-xs text-blue-500 flex items-center gap-1">
                    <Image size={12} />
                    {r.images.length} 张图片
                  </span>
                )}
                <Badge variant="secondary" className="ml-auto text-xs">
                  {(r.score * 100).toFixed(1)}%
                </Badge>
              </div>
              <p className="text-sm text-gray-600 leading-relaxed line-clamp-2">
                {r.highlight || r.content}
              </p>
              <div className="mt-2 text-xs text-blue-500 flex items-center gap-1">
                <Info size={12} />
                <span>点击查看详情</span>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* 详情弹窗 */}
      <Dialog open={!!selectedResult} onClose={() => setSelectedResult(null)}>
        {selectedResult && (
          <div className="max-w-3xl max-h-[85vh] overflow-y-auto">
            <div className="flex items-center gap-2 mb-4 pb-3 border-b">
              <FileText size={18} />
              <h2 className="text-lg font-semibold">{selectedResult.document_name}</h2>
            </div>

            <div className="space-y-4">
              {/* 元数据 */}
              <div className="grid grid-cols-2 gap-3 text-sm">
                <div className="bg-gray-50 rounded-lg p-3">
                  <span className="text-gray-500">相关性分数</span>
                  <p className="font-medium mt-1">{(selectedResult.score * 100).toFixed(1)}%</p>
                </div>
                <div className="bg-gray-50 rounded-lg p-3">
                  <span className="text-gray-500">块类型</span>
                  <p className="font-medium mt-1">{selectedResult.chunk_type || 'text'}</p>
                </div>
                <div className="bg-gray-50 rounded-lg p-3">
                  <span className="text-gray-500">语言</span>
                  <p className="font-medium mt-1">{selectedResult.language || 'unknown'}</p>
                </div>
                {selectedResult.page_number && (
                  <div className="bg-gray-50 rounded-lg p-3">
                    <span className="text-gray-500">页码</span>
                    <p className="font-medium mt-1">第 {selectedResult.page_number} 页</p>
                  </div>
                )}
              </div>

              {/* 文本内容 */}
              <div>
                <h4 className="text-sm font-medium text-gray-700 mb-2">内容</h4>
                <div className="bg-gray-50 rounded-lg p-4 text-sm text-gray-700 leading-relaxed whitespace-pre-wrap">
                  {selectedResult.content}
                </div>
              </div>

              {/* 图片 */}
              {selectedResult.images && selectedResult.images.length > 0 && (
                <div>
                  <h4 className="text-sm font-medium text-gray-700 mb-2 flex items-center gap-2">
                    <Image size={16} />
                    相关图片 ({selectedResult.images.length})
                  </h4>
                  <div className="grid gap-3">
                    {selectedResult.images.map((img, idx) => (
                      <div key={idx} className="border rounded-lg overflow-hidden">
                        <img
                          src={img.url}
                          alt={img.caption || `图片 ${idx + 1}`}
                          className="w-full h-auto max-h-80 object-contain bg-gray-100"
                        />
                        {img.caption && (
                          <p className="text-xs text-gray-500 p-2 bg-gray-50">{img.caption}</p>
                        )}
                        <p className="text-xs text-gray-400 p-2 truncate">{img.path}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* 高亮片段 */}
              {selectedResult.highlight && (
                <div>
                  <h4 className="text-sm font-medium text-gray-700 mb-2">匹配高亮</h4>
                  <div
                    className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 text-sm leading-relaxed"
                    dangerouslySetInnerHTML={{ __html: selectedResult.highlight }}
                  />
                </div>
              )}
            </div>
          </div>
        )}
      </Dialog>
    </div>
  )
}
