import * as React from "react"
import { cn } from "@/utils/cn"

interface AvatarProps {
  src?: string | null
  name?: string
  size?: "sm" | "md" | "lg"
  className?: string
}

export function Avatar({ src, name, size = "md", className }: AvatarProps) {
  const sizeClasses = {
    sm: "h-8 w-8 text-xs",
    md: "h-10 w-10 text-sm",
    lg: "h-12 w-12 text-base",
  }

  if (src) {
    return (
      <img
        src={src}
        alt={name || "avatar"}
        className={cn(
          "rounded-full object-cover",
          sizeClasses[size],
          className
        )}
      />
    )
  }

  const initial = (name || "U").charAt(0).toUpperCase()

  return (
    <div
      className={cn(
        "rounded-full bg-blue-100 text-blue-600 flex items-center justify-center font-medium",
        sizeClasses[size],
        className
      )}
    >
      {initial}
    </div>
  )
}
